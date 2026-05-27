// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * AWQ 4-bit GEMM kernel for Volta (SM 7.0) with online dequantization.
// * Computes C = A (fp16) @ dequant(B int4) using Volta WMMA m16n16k16.
// *
// * 1. TILE SHAPE:     M=16 per block, N={64,128} (2 warps x N/2 each), K=32.
// * 2. WMMA OP:        wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32
// * 3. LOADS:          ld.shared.v4.u32 (A, row-major) / ld.shared.v2.u32 (B, row-major).
// * 4. DEQUANTIZATION: Inline via dequantize_s4_to_fp16x2() + scale/zero FMA + prmt.b32 nibble reorder.
// * 5. SPLIT-K:        Supported via blockIdx.z reduction in outer Python wrapper (fp32 buffer).
// ======================================================================================
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "dequantize.cuh"

namespace vllm {
namespace awq {

// ======================================================================================
// GEMM KERNEL: m16n16k16 (X in {64, 128}) with inline 4-bit dequantization
// ======================================================================================
// Grid mapping:
//   blockIdx.x  =  M_tiles * N_tiles * split_k_iters  (flattened 3D index)
//   blockIdx.y  =  1                                   (unused)
//   blockIdx.z  =  1                                   (unused)
// Thread mapping:
//   threadIdx.x =  [0..31]   lane within warp
//   threadIdx.y =  [0..1]    warp id (each warp handles N/2 output columns)
// ======================================================================================
template <int N>
__global__ void __launch_bounds__(64)
gemm_forward_4bit_cuda_m16n16k16(
    const int    G,                // Group size for quantization scales
    const int    split_k_iters,    // Number of K splits (outer reduction dim)
    half*  __restrict__ A,         // [M, IC]   fp16 input activations
    int*   __restrict__ B,         // [IC, OC/8] int32 packed int4 weights
    half*  __restrict__ scales,    // [IC/G, OC] fp16 quantization scales
    int*   __restrict__ zeros,     // [IC/G, OC/8] int32 packed int4 zero-points
    const int    M,                // Number of input rows
    const int    IC,               // Input channels  (K dim)
    const int    OC,               // Output channels (N dim)
    float* __restrict__ C          // [split_k, M, OC] fp32 output (for precise Split-K reduction)
) {
    static_assert(N == 64 || N == 128, "Only cta_N = 64 or 128 is supported");

    // ==========================================================================
    // Constants
    // ==========================================================================
    static constexpr uint32_t ZERO_MASK       = 0x0;
    static constexpr int      SMEM_A_PAD      = 8;      // Pad to avoid bank conflicts
    static constexpr int      SMEM_B_PAD      = 8;
    static constexpr int      ROW_STRIDE_WARP = 8;      // Rows per warp for A load
    static constexpr int      ROW_STRIDE_B    = 2 * 32 * 8 / N;

    // ==========================================================================
    // Init: thread/warp/lane IDs and block indices
    // ==========================================================================
    const int tid     = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int lane_id = tid;

    const int j_tiles   = (OC + N - 1) / N;
    const int m_tiles   = (M + 16 - 1) / 16;
    const int blockIdx_y = blockIdx.x % (m_tiles * j_tiles);
    const int blockIdx_z = blockIdx.x / (m_tiles * j_tiles);

    const int tile_m = blockIdx_y / j_tiles;
    const int tile_n = blockIdx_y % j_tiles;

    // ==========================================================================
    // Init: shared memory buffers (padded to avoid bank conflicts)
    //   A_shared: [16, 32 + 8] halves   = 1280 bytes
    //   B_shared: [32, N + 8]  halves   = 32*(N+8)*2 bytes
    // ==========================================================================
    __shared__ half A_shared[16 * (32 + SMEM_A_PAD)];
    __shared__ half B_shared[32 * (N  + SMEM_B_PAD)];

    // ==========================================================================
    // Init: register-level fragment buffers
    //   A_shared_warp: 8 x uint32  = 16 halves  (one row-major m16n16k16 A fragment)
    //   B_shared_warp: (N/32)*8 uint32 = N/4 halves per warp (one B fragment per j_tile)
    //   C_warp:        (N/32)*8 floats = N/4 floats (accumulators)
    // ==========================================================================
    uint32_t A_shared_warp[8];
    uint32_t B_shared_warp[N / 4];
    float    C_warp[N / 4];

    for (int j_init = 0; j_init < N / 32; ++j_init) {
        for (int i = 0; i < 8; ++i) {
            C_warp[j_init * 8 + i] = 0.0f;
        }
    }

    // ==========================================================================
    // Init: global memory pointers with per-thread offsets
    // ==========================================================================
    const bool valid_a_row = (tile_m * 16 + warp_id * ROW_STRIDE_WARP + tid / 4) < M;

    half* A_ptr = A
        + (tile_m * 16 + warp_id * ROW_STRIDE_WARP + tid / 4) * IC
        + (tid % 4) * 8;

    int* B_ptr = B
        + warp_id * (OC / 8) * (256 / N)
        + (tid / (N / 8)) * (OC / 8)
        + tile_n * (N / 8)
        + (tid % (N / 8));

    half* A_shared_ptr = A_shared
        + warp_id * ROW_STRIDE_WARP * (32 + SMEM_A_PAD)
        + (tid / 4) * (32 + SMEM_A_PAD)
        + (tid % 4) * 8;

    half* B_shared_ptr = B_shared
        + warp_id * (ROW_STRIDE_B / 2) * (N + SMEM_B_PAD)
        + (tid / (N / 8)) * (N + SMEM_B_PAD)
        + (tid % (N / 8)) * 8;

    int*  zeros_ptr  = zeros  + tile_n * (N / 8) + (tid % (N / 8));
    half* scales_ptr = scales + tile_n * N        + (tid % (N / 8)) * 8;

    float* C_ptr_base = C
        + static_cast<long long>(blockIdx_z) * M * OC
        + tile_n * N;

    // ==========================================================================
    // Init: K-loop bounds for split-K reduction
    // ==========================================================================
    int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
    if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) {
        k_bound -= 1;
    }

    // ==========================================================================
    // MAIN LOOP: iterate over K tiles (each tile = 32 elements along IC)
    // ==========================================================================
    for (int k_iter = 0; k_iter < k_bound; ++k_iter) {
        const int k_tile = k_iter * split_k_iters + blockIdx_z;

        __syncthreads();

        // ======================================================================
        // Load: A tile (16 x 32) from global -> shared memory
        // Layout:  A[row, col] -> A_shared[row, col + padding]
        // ======================================================================
        if (valid_a_row) {
            *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_tile * 32);
        } else {
            *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
        }

        // ======================================================================
        // Load: B scale and zero-point for current K group qzeros interleaved inside each int32
        // dequantize_s4_to_fp16x2: .x={z0,z4}, .y={z1,z5}, .z={z2,z6}, .w={z3,z7}
        // We must permute:         .x={z0,z1}, .y={z2,z3}, .z={z4,z5}, .w={z6,z7}
        // ======================================================================
        const uint32_t zeros_raw  = *(uint32_t*)(zeros_ptr  + (k_tile * 32 / G) * (OC / 8));
        uint4 zero_fp16  = dequantize_s4_to_fp16x2(zeros_raw);

        uint4 zero_fp16_seq;
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.x) : "r"(zero_fp16.x), "r"(zero_fp16.z));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.y) : "r"(zero_fp16.x), "r"(zero_fp16.z));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.z) : "r"(zero_fp16.y), "r"(zero_fp16.w));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.w) : "r"(zero_fp16.y), "r"(zero_fp16.w));
        zero_fp16 = zero_fp16_seq;

        const uint4    scale_fp16 = *(uint4*)(scales_ptr + (k_tile * 32 / G) * OC);

        int* B_ptr_local = B_ptr + k_tile * 32 * (OC / 8);

        // ======================================================================
        // Load: B tile from global -> dequantize -> shared memory
        //   Each thread loads 1 int32 (8 int4s), converts to 8 fp16, applies
        //   (w - zero) * scale, writes back as uint4 (128-bit).
        //   qweight uses the same interleaved nibble packing as qzeros, so we
        //   permute B_deq with the same prmt.b32 pattern before subtracting zero.
        // ======================================================================
        for (int tile_b = 0; tile_b < N / 16; ++tile_b) {
            const uint32_t B_packed = *(uint32_t*)(B_ptr_local + tile_b * ROW_STRIDE_B * (OC / 8));
            uint4 B_deq = dequantize_s4_to_fp16x2(B_packed);

            // Reorder interleaved AWQ qweight nibbles to sequential layout
            uint4 B_deq_seq;
            asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(B_deq_seq.x) : "r"(B_deq.x), "r"(B_deq.z));
            asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(B_deq_seq.y) : "r"(B_deq.x), "r"(B_deq.z));
            asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(B_deq_seq.z) : "r"(B_deq.y), "r"(B_deq.w));
            asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(B_deq_seq.w) : "r"(B_deq.y), "r"(B_deq.w));
            B_deq = B_deq_seq;

            // (w - zero) * scale  -- applied per fp16x2 pair
            asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.x) : "r"(B_deq.x), "r"(zero_fp16.x));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.x) : "r"(B_deq.x), "r"(scale_fp16.x), "r"(ZERO_MASK));
            asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.y) : "r"(B_deq.y), "r"(zero_fp16.y));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.y) : "r"(B_deq.y), "r"(scale_fp16.y), "r"(ZERO_MASK));
            asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.z) : "r"(B_deq.z), "r"(zero_fp16.z));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.z) : "r"(B_deq.z), "r"(scale_fp16.z), "r"(ZERO_MASK));
            asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.w) : "r"(B_deq.w), "r"(zero_fp16.w));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.w) : "r"(B_deq.w), "r"(scale_fp16.w), "r"(ZERO_MASK));

            *(uint4*)(B_shared_ptr + tile_b * ROW_STRIDE_B * (N + SMEM_B_PAD)) = B_deq;
        }

        __syncthreads();

        // ======================================================================
        // Compute: split K=32 tile into two K=16 sub-tiles and run WMMA mma
        // ======================================================================
        for (int k_sub = 0; k_sub < 2; ++k_sub) {

            // ==================================================================
            // Load: A fragment (m16n16k16 ROW MAJOR) from A_shared
            // Lane mapping:  r_base = (lid&3) + ((lid>>4)&1)*4 + ((lid>>2)&1)*8
            // Access:        2x ld.shared.v4.u32 (128-bit vectorized loads)
            // ==================================================================
            const unsigned smem_addr = __cvta_generic_to_shared(&A_shared[k_sub * 16]);
            asm volatile(
                "{\n\t"
                ".reg .u32 lid, r_base, base, t;\n\t"
                "mov.u32    lid, %%laneid;\n\t"
                "and.b32    r_base, lid, 3;\n\t"
                "shr.b32    t, lid, 4; and.b32 t, t, 1; mad.lo.u32 r_base, t, 4, r_base;\n\t"
                "shr.b32    t, lid, 2; and.b32 t, t, 1; mad.lo.u32 r_base, t, 8, r_base;\n\t"
                "mul.lo.u32 base, r_base, %9;\n\t"
                "shl.b32    base, base, 1;\n\t"
                "add.u32    base, base, %8;\n\t"
                "ld.shared.v4.u32 {%0, %1, %2, %3}, [base];\n\t"
                "ld.shared.v4.u32 {%4, %5, %6, %7}, [base+16];\n\t"
                "}"
                : "=r"(A_shared_warp[0]), "=r"(A_shared_warp[1]),
                  "=r"(A_shared_warp[2]), "=r"(A_shared_warp[3]),
                  "=r"(A_shared_warp[4]), "=r"(A_shared_warp[5]),
                  "=r"(A_shared_warp[6]), "=r"(A_shared_warp[7])
                : "r"(smem_addr), "r"(32 + SMEM_A_PAD)
                : "memory"
            );

            // ==================================================================
            // Load: B fragments (one per N/32 j_tiles) from B_shared (ROW MAJOR)
            // Lane mapping:  r_base = lid & 3
            //                c_base = ((lid>>3)&1)*8 + ((lid>>4)&1)*4
            // Access:        4x ld.shared.v2.u32 (strided rows)
            // ==================================================================
            for (int n_tile = 0; n_tile < N / 32; ++n_tile) {
                const unsigned smem_addr = __cvta_generic_to_shared(
                    &B_shared[k_sub * (N * 16 + 128) + warp_id * (N / 2) + n_tile * 16]);

                asm volatile(
                    "{\n\t"
                    ".reg .u32 lid, r_base, c_base, base, stride, a0, a1, a2, a3, t1, t2;\n\t"
                    "mov.u32    lid, %%laneid;\n\t"
                    "and.b32    r_base, lid, 3;\n\t"
                    "shr.b32    t1, lid, 3; and.b32 t1, t1, 1; shl.b32 t1, t1, 3;\n\t"
                    "shr.b32    t2, lid, 4; and.b32 t2, t2, 1; shl.b32 t2, t2, 2;\n\t"
                    "add.u32    c_base, t1, t2;\n\t"
                    "mad.lo.u32 base, r_base, %9, c_base;\n\t"
                    "shl.b32    base, base, 1;\n\t"
                    "add.u32    base, base, %8;\n\t"
                    "shl.b32    stride, %9, 3;\n\t"
                    "mov.u32    a0, base;\n\t"
                    "add.u32    a1, base, stride;\n\t"
                    "add.u32    a2, a1, stride;\n\t"
                    "add.u32    a3, a2, stride;\n\t"
                    "ld.shared.v2.u32 {%0, %1}, [a0];\n\t"
                    "ld.shared.v2.u32 {%2, %3}, [a1];\n\t"
                    "ld.shared.v2.u32 {%4, %5}, [a2];\n\t"
                    "ld.shared.v2.u32 {%6, %7}, [a3];\n\t"
                    "}"
                    : "=r"(B_shared_warp[n_tile * 8 + 0]), "=r"(B_shared_warp[n_tile * 8 + 1]),
                      "=r"(B_shared_warp[n_tile * 8 + 2]), "=r"(B_shared_warp[n_tile * 8 + 3]),
                      "=r"(B_shared_warp[n_tile * 8 + 4]), "=r"(B_shared_warp[n_tile * 8 + 5]),
                      "=r"(B_shared_warp[n_tile * 8 + 6]), "=r"(B_shared_warp[n_tile * 8 + 7])
                    : "r"(smem_addr), "r"(N + SMEM_B_PAD)
                    : "memory"
                );
            }

            // ==================================================================
            // Compute: WMMA m16n16k16 row.row f32 accumulation
            //   C_warp[j_tile * 8 + 0..7] += A_shared_warp @ B_shared_warp[j_tile]
            // ==================================================================
            for (int j_tile = 0; j_tile < N / 32; ++j_tile) {
                asm volatile(
                    "wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7}, "
                    "{%8,%9,%10,%11,%12,%13,%14,%15}, "
                    "{%16,%17,%18,%19,%20,%21,%22,%23}, "
                    "{%24,%25,%26,%27,%28,%29,%30,%31};\n"
                    : "=f"(C_warp[j_tile * 8 + 0]), "=f"(C_warp[j_tile * 8 + 1]),
                      "=f"(C_warp[j_tile * 8 + 2]), "=f"(C_warp[j_tile * 8 + 3]),
                      "=f"(C_warp[j_tile * 8 + 4]), "=f"(C_warp[j_tile * 8 + 5]),
                      "=f"(C_warp[j_tile * 8 + 6]), "=f"(C_warp[j_tile * 8 + 7])
                    : "r"(A_shared_warp[0]),          "r"(A_shared_warp[1]),
                      "r"(A_shared_warp[2]),          "r"(A_shared_warp[3]),
                      "r"(A_shared_warp[4]),          "r"(A_shared_warp[5]),
                      "r"(A_shared_warp[6]),          "r"(A_shared_warp[7]),
                      "r"(B_shared_warp[j_tile * 8 + 0]), "r"(B_shared_warp[j_tile * 8 + 1]),
                      "r"(B_shared_warp[j_tile * 8 + 2]), "r"(B_shared_warp[j_tile * 8 + 3]),
                      "r"(B_shared_warp[j_tile * 8 + 4]), "r"(B_shared_warp[j_tile * 8 + 5]),
                      "r"(B_shared_warp[j_tile * 8 + 6]), "r"(B_shared_warp[j_tile * 8 + 7]),
                      "f"(C_warp[j_tile * 8 + 0]),    "f"(C_warp[j_tile * 8 + 1]),
                      "f"(C_warp[j_tile * 8 + 2]),    "f"(C_warp[j_tile * 8 + 3]),
                      "f"(C_warp[j_tile * 8 + 4]),    "f"(C_warp[j_tile * 8 + 5]),
                      "f"(C_warp[j_tile * 8 + 6]),    "f"(C_warp[j_tile * 8 + 7])
                );
            }
        }
    }

    // ==========================================================================
    // Store: Write accumulated C_warp back to global memory
    //   Volta m16n16k16 row_major accumulator layout:
    //     r_base = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
    //     c_base = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
    //   8 elements per lane arranged as 2x2 blocks at offsets (r, c), (r, c+1),
    //   (r+2, c), (r+2, c+1), (r, c+4), (r, c+5), (r+2, c+4), (r+2, c+5).
    // ==========================================================================
    const int r0 = ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4 + (lane_id & 1);
    const int c0 = ((lane_id >> 3) & 1) * 8 + ((lane_id >> 1) & 1) * 2;

    for (int n_out = 0; n_out < (N / 32); ++n_out) {
        for (int frag_idx = 0; frag_idx < 8; ++frag_idx) {
            int r_off, c_off;
            switch (frag_idx) {
                case 0: r_off = r0;     c_off = c0;     break;
                case 1: r_off = r0;     c_off = c0 + 1; break;
                case 2: r_off = r0 + 2; c_off = c0;     break;
                case 3: r_off = r0 + 2; c_off = c0 + 1; break;
                case 4: r_off = r0;     c_off = c0 + 4; break;
                case 5: r_off = r0;     c_off = c0 + 5; break;
                case 6: r_off = r0 + 2; c_off = c0 + 4; break;
                case 7: r_off = r0 + 2; c_off = c0 + 5; break;
                default: r_off = 0;     c_off = 0;      break;
            }

            const int global_row = tile_m * 16 + r_off;
            const int global_col = n_out * 16 + c_off + warp_id * (N / 2);

            if (global_row < M) {
                C_ptr_base[global_row * OC + global_col] = C_warp[n_out * 8 + frag_idx];
            }
        }
    }
}

// ======================================================================================
// DEQUANTIZE KERNEL: unpack int4 weights to fp16 with scale/zero applied
// ======================================================================================
__global__ void __launch_bounds__(64)
dequantize_weights(
    int*  __restrict__ B,          // [IC, OC/8] packed int4 weights
    half* __restrict__ scales,     // [IC/G, OC] fp16 scales
    int*  __restrict__ zeros,      // [IC/G, OC/8] packed int4 zero-points
    half* __restrict__ C,          // [IC, OC]   fp16 dequantized output
    const int G                    // Group size
) {
    static constexpr uint32_t ZERO_MASK = 0x0;

    const int N   = blockDim.x * gridDim.x;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    half* C_ptr2      = C      + 8 * col + 8 * row * N;
    int*  B_ptr2      = B      + col + row * N;
    int*  zeros_ptr2  = zeros  + col + (row / G) * N;
    half* scales_ptr2 = scales + 8 * col + (row / G) * N * 8;

    const uint32_t zeros_raw  = *(uint32_t*)(zeros_ptr2);
    uint4 zero_fp16  = dequantize_s4_to_fp16x2(zeros_raw);

    // Reorder interleaved AWQ qzeros nibbles to sequential layout
    uint4 zero_fp16_seq;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.x) : "r"(zero_fp16.x), "r"(zero_fp16.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.y) : "r"(zero_fp16.x), "r"(zero_fp16.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.z) : "r"(zero_fp16.y), "r"(zero_fp16.w));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.w) : "r"(zero_fp16.y), "r"(zero_fp16.w));
    zero_fp16 = zero_fp16_seq;

    const uint4    scale_fp16 = *(uint4*)(scales_ptr2);
    const uint32_t B_packed   = *(uint32_t*)B_ptr2;
    uint4          B_deq      = dequantize_s4_to_fp16x2(B_packed);

    // Reorder interleaved AWQ qweight nibbles to sequential layout
    uint4 B_deq_seq;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(B_deq_seq.x) : "r"(B_deq.x), "r"(B_deq.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(B_deq_seq.y) : "r"(B_deq.x), "r"(B_deq.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(B_deq_seq.z) : "r"(B_deq.y), "r"(B_deq.w));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(B_deq_seq.w) : "r"(B_deq.y), "r"(B_deq.w));
    B_deq = B_deq_seq;

    // (w - zero) * scale
    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.x) : "r"(B_deq.x), "r"(zero_fp16.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.x) : "r"(B_deq.x), "r"(scale_fp16.x), "r"(ZERO_MASK));
    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.y) : "r"(B_deq.y), "r"(zero_fp16.y));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.y) : "r"(B_deq.y), "r"(scale_fp16.y), "r"(ZERO_MASK));
    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.z) : "r"(B_deq.z), "r"(zero_fp16.z));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.z) : "r"(B_deq.z), "r"(scale_fp16.z), "r"(ZERO_MASK));
    asm volatile("sub.f16x2    %0, %1, %2;\n" : "=r"(B_deq.w) : "r"(B_deq.w), "r"(zero_fp16.w));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_deq.w) : "r"(B_deq.w), "r"(scale_fp16.w), "r"(ZERO_MASK));

    // Direct 128-bit store (C_ptr2 guaranteed 16-byte aligned by construction)
    *(uint4*)C_ptr2 = B_deq;
}

} // namespace awq
} // namespace vllm

// ======================================================================================
// LAUNCHER: awq_dequantize
// ======================================================================================
torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int64_t       split_k_iters,
    int64_t       thx,
    int64_t       thy
) {
    const int in_c  = _kernel.size(0);
    const int qout_c = _kernel.size(1);
    const int out_c = qout_c * 8;
    const int G     = in_c / _scaling_factors.size(0);

    int x_thread = thx;
    int y_thread = thy;
    int x_blocks = 1;
    int y_blocks = 1;
    if (thx == 0) x_thread = qout_c;
    if (thy == 0) y_thread = in_c;
    if (thx == 0 && thy == 0) {
        x_thread = 8;
        y_thread = 8;
        x_blocks = static_cast<int>(qout_c / 8);
        y_blocks = static_cast<int>(in_c / 8);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));
    auto options = torch::TensorOptions()
                       .dtype(_scaling_factors.dtype())
                       .device(_scaling_factors.device());
    at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

    auto kernel          = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto de_kernel       = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros           = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    dim3 num_blocks(x_blocks, y_blocks);
    dim3 threads_per_block(x_thread, y_thread);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    vllm::awq::dequantize_weights<<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);

    return _de_kernel;
}

// ======================================================================================
// LAUNCHER: awq_gemm
// ======================================================================================
// in_feats:         [M, IC]      fp16
// kernel:           [IC, OC/8]   int32 (packed int4)
// scaling_factors:  [IC/G, OC]   fp16
// zeros:            [IC/G, OC/8] int32 (packed int4)
// split_k_iters:    K-split factor (final result reduced via .sum(0) in fp32)
//
// Returns: [M, OC] fp16 (cast from fp32 intermediate to preserve Split-K precision)
// ======================================================================================
torch::Tensor awq_gemm(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int64_t       split_k_iters
) {
    const int num_in_feats    = _in_feats.size(0);
    const int num_in_channels = _in_feats.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(_in_feats.device());
    at::Tensor _out_feats =
        torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);

    const int num_out_feats   = _out_feats.size(-2);
    const int num_out_channels = _out_feats.size(-1);

    auto in_feats        = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel          = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats       = _out_feats.data_ptr<float>();
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros           = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    const int group_size = num_in_channels / _scaling_factors.size(0);

    // ==========================================================================
    // Shape validation
    // ==========================================================================
    TORCH_CHECK(num_out_channels % 64 == 0,  "OC must be a multiple of cta_N = 64");
    TORCH_CHECK(num_out_channels % 8 == 0,   "OC must be a multiple of pack_num = 8");
    TORCH_CHECK(group_size % 32 == 0,        "Group size must be a multiple of 32");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ==========================================================================
    // Dispatch: choose cta_N = 128 or 64 based on OC
    // ==========================================================================
    if (num_out_channels % 128 == 0) {
        const int j_tiles = num_out_channels / 128;
        dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_tiles * split_k_iters);
        dim3 threads_per_block(32, 2);
        vllm::awq::gemm_forward_4bit_cuda_m16n16k16<128>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                group_size, split_k_iters,
                in_feats, kernel, scaling_factors, zeros,
                num_in_feats, num_in_channels, num_out_channels, out_feats);
    } else if (num_out_channels % 64 == 0) {
        const int j_tiles = num_out_channels / 64;
        dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_tiles * split_k_iters);
        dim3 threads_per_block(32, 2);
        vllm::awq::gemm_forward_4bit_cuda_m16n16k16<64>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                group_size, split_k_iters,
                in_feats, kernel, scaling_factors, zeros,
                num_in_feats, num_in_channels, num_out_channels, out_feats);
    }

    return _out_feats.sum(0).to(_in_feats.dtype());
}
