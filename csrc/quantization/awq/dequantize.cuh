// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * AWQ 4-bit dequantization kernel for Volta (SM 7.0).
// * Converts packed int4 weights into fp16x2 using pure PTX bit-tricks.
// *
// * 1. PACKING FORMAT: 8 x int4 (unsigned 0..15) packed into single uint32.
// * 2. PIPELINE STAGES: Extract -> Convert (sub/fma magic) -> Permute (sequential).
// * 3. MAGIC CONSTANTS: Exploit fp16 exponent bias (1024) for O(1) int->fp conversion.
// * 4. OUTPUT ORDER:    Sequential {e0,e1}, {e2,e3}, {e4,e5}, {e6,e7} via prmt.b32.
// ======================================================================================
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {
namespace awq {

// ======================================================================================
// DEQUANTIZE: int4 (packed) -> fp16x2 (sequential)
// ======================================================================================
// Input:   1 x uint32 holding 8 unsigned 4-bit integers [e0..e7]
// Output:  uint4 (= 4 x uint32 = 8 x half) in sequential element order:
//            .x = {e0, e1}    .y = {e2, e3}    .z = {e4, e5}    .w = {e6, e7}
//
// Stage 1 (EXTRACT):  lop3.b32 builds {1024+value} fp16 pairs from raw int4 nibbles.
// Stage 2 (CONVERT):  sub/fma magic subtracts bias to get actual fp16 values.
// Stage 3 (PERMUTE):  prmt.b32 reshuffles interleaved halves into sequential order.
// ======================================================================================
__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
    // ==============================================================================
    // Stage 1: EXTRACT int4 nibbles and inject fp16 exponent bias (1024 = 0x6400).
    // Result layout after lop3:
    //   h[0] = {e0, e4}   (bottom nibbles of low  / high halves)
    //   h[1] = {e1, e5}   (top nibbles    of low  / high halves)
    //   h[2] = {e2, e6}   (bottom nibbles of high / high-high halves)
    //   h[3] = {e3, e7}   (top nibbles    of high / high-high halves)
    // ==============================================================================
    uint32_t h[4];
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t immLut              = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK         = 0x000f000f;
    static constexpr uint32_t TOP_MASK            = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC   = 0x64006400;  // {1024, 1024}

    const uint32_t top_i4s = i4s >> 8;

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0]) : "r"(i4s),      "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1]) : "r"(i4s),      "n"(TOP_MASK),    "n"(I4s_TO_F16s_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2]) : "r"(top_i4s),  "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3]) : "r"(top_i4s),  "n"(TOP_MASK),    "n"(I4s_TO_F16s_MAGIC), "n"(immLut));

    // ==============================================================================
    // Stage 2: CONVERT biased fp16 pairs into actual values.
    //   elt_01, elt_45:  value = biased - 1024              (via sub.f16x2)
    //   elt_23, elt_67:  value = biased * (1/16) + (-64)    (via fma.rn.f16x2)
    // ==============================================================================
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;  // {1024, 1024}
    static constexpr uint32_t ONE_SIXTEENTH      = 0x2c002c00;  // {1/16, 1/16}
    static constexpr uint32_t NEG_64             = 0xd400d400;  // {-64,  -64}

    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    // ==============================================================================
    // Stage 3: PERMUTE interleaved halves into sequential element order.
    //   prmt.b32 dst, src_a, src_b, sel   (selects 4 bytes out of 8 = src_b:src_a)
    //   sel = 0x5410 -> lower halves  of both src_a and src_b
    //   sel = 0x7632 -> upper halves  of both src_a and src_b
    // Result: {e0,e1}, {e2,e3}, {e4,e5}, {e6,e7}
    // ==============================================================================
    uint4 result;
    uint32_t* r = reinterpret_cast<uint32_t*>(&result);

    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(r[0]) : "r"(h[0]), "r"(h[1]));  // {e0, e1}
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(r[1]) : "r"(h[2]), "r"(h[3]));  // {e2, e3}
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(r[2]) : "r"(h[0]), "r"(h[1]));  // {e4, e5}
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(r[3]) : "r"(h[2]), "r"(h[3]));  // {e6, e7}

    return result;
}

} // namespace awq
} // namespace vllm