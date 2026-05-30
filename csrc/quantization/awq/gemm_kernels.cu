// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007
// * SPDX-License: BSD-3-Clause
// ======================================================================================
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "dequantize.cuh"

namespace vllm {
namespace awq {

// ======================================================================================
// GEMM KERNEL: m16n16k16 (N in {64, 128}) with inline 4-bit dequantization
// ======================================================================================
// Grid mapping:
//   blockIdx.x  =  M_tiles * N_tiles * split_k_iters  (flattened 3D index)
//   blockIdx.y  =  1                                   (unused)
//   blockIdx.z  =  1                                   (unused)
// Thread mapping:
//   threadIdx.x =  [0..31]   lane within warp
//   threadIdx.y =  [0..1]    warp id (each warp handles N/2 output columns)
// ======================================================================================
template <int N, bool SPLIT_K>
__global__ void __launch_bounds__(64, 8)
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
    void*  __restrict__ C_void     // half* if !SPLIT_K, float* if SPLIT_K
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

    #pragma unroll  // OPT#5: explicit unroll for ILP
    for (int j_init = 0; j_init < N / 32; ++j_init) {
        #pragma unroll
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

    // ==========================================================================
    // Init: K-loop bounds for split-K reduction
    // ==========================================================================
    int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
    if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) {
        k_bound -= 1;
    }
    k_bound = max(1, k_bound);

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
        // Load: zeros (int4 packed) via ld.global.nc (bypasses L1 cache)
        // Then: single-pass dequantize to interleaved fp16x2 (OPT#1 + OPT#4)
        // ======================================================================
        uint32_t zeros_raw;
        {
            uint64_t zeros_addr = reinterpret_cast<uint64_t>(
                zeros_ptr + (k_tile * 32 / G) * (OC / 8));
            asm volatile("ld.global.nc.u32 %0, [%1];\n"
                : "=r"(zeros_raw) : "l"(zeros_addr) : "memory");
        }
        uint4 zero_fp16 = dequantize_s4_to_fp16x2_interleaved(zeros_raw);

        // ======================================================================
        // Load: scales (fp16) via ld.global.nc.v4.b32 (OPT#4)
        // ======================================================================
        uint4 scale_fp16;
        {
            uint64_t scales_addr = reinterpret_cast<uint64_t>(
                scales_ptr + (k_tile * 32 / G) * OC);
            asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(scale_fp16.x), "=r"(scale_fp16.y),
                  "=r"(scale_fp16.z), "=r"(scale_fp16.w)
                : "l"(scales_addr) : "memory");
        }

        int* B_ptr_local = B_ptr + k_tile * 32 * (OC / 8);

        // ======================================================================
        // Load: B tile from global -> dequantize -> shared memory
        //   Each thread loads 1 int32 (8 int4s), converts to 8 fp16, applies
        //   (w - zero) * scale, writes back as uint4 (128-bit).
        //   Interleaved layout preserved throughout — no prmt.b32 needed.
        // ======================================================================
        #pragma unroll  // OPT#5
        for (int tile_b = 0; tile_b < N / 16; ++tile_b) {
            uint32_t B_packed;
            {
                uint64_t b_addr = reinterpret_cast<uint64_t>(
                    B_ptr_local + tile_b * ROW_STRIDE_B * (OC / 8));
                asm volatile("ld.global.nc.u32 %0, [%1];\n"
                    : "=r"(B_packed) : "l"(b_addr) : "memory");
            }

            uint4 B_deq = dequantize_s4_to_fp16x2_interleaved(B_packed);

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
        #pragma unroll  // OPT#5
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
            #pragma unroll  // OPT#5
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
            #pragma unroll  // OPT#5
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

    if constexpr (!SPLIT_K) {
        // ======================================================================
        // FAST PATH: Direct fp16 output (no reduction needed)
        // Pack 2 floats into half2, then store as single uint32_t (4 bytes).
        // ======================================================================
        half* C = reinterpret_cast<half*>(C_void);
        half* C_ptr_base = C + tile_n * N;

        #pragma unroll
        for (int n_out = 0; n_out < (N / 32); ++n_out) {
            int global_row_0 = tile_m * 16 + r0;
            int global_row_2 = global_row_0 + 2;
            int global_col_0 = n_out * 16 + c0 + warp_id * (N / 2);
            int global_col_4 = global_col_0 + 4;

            if (global_row_0 < M) {
                half2 h01 = __float22half2_rn(make_float2(C_warp[n_out * 8 + 0], C_warp[n_out * 8 + 1]));
                half2 h45 = __float22half2_rn(make_float2(C_warp[n_out * 8 + 4], C_warp[n_out * 8 + 5]));
                *reinterpret_cast<uint32_t*>(&C_ptr_base[global_row_0 * OC + global_col_0]) = *reinterpret_cast<uint32_t*>(&h01);
                *reinterpret_cast<uint32_t*>(&C_ptr_base[global_row_0 * OC + global_col_4]) = *reinterpret_cast<uint32_t*>(&h45);
            }
            if (global_row_2 < M) {
                half2 h23 = __float22half2_rn(make_float2(C_warp[n_out * 8 + 2], C_warp[n_out * 8 + 3]));
                half2 h67 = __float22half2_rn(make_float2(C_warp[n_out * 8 + 6], C_warp[n_out * 8 + 7]));
                *reinterpret_cast<uint32_t*>(&C_ptr_base[global_row_2 * OC + global_col_0]) = *reinterpret_cast<uint32_t*>(&h23);
                *reinterpret_cast<uint32_t*>(&C_ptr_base[global_row_2 * OC + global_col_4]) = *reinterpret_cast<uint32_t*>(&h67);
            }
        }
    } else {
        // ======================================================================
        // SPLIT-K PATH: Write fp32 partial sums for later reduction.
        // Scalar stores acceptable here since reduction kernel aggregates.
        // ======================================================================
        float* C = reinterpret_cast<float*>(C_void);
        float* C_ptr_base = C + static_cast<long long>(blockIdx_z) * M * OC + tile_n * N;

        #pragma unroll
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
}

__global__ void __launch_bounds__(256)
split_k_reduce_kernel(
    const float* __restrict__ C_split,
    half* __restrict__ C_out,
    int M, int OC, int split_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < OC) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < split_k; ++i) {
            sum += C_split[i * M * OC + row * OC + col];
        }
        C_out[row * OC + col] = __float2half(sum);
    }
}

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

    // Reorder interleaved AWQ qzeros nibbles to sequential layout (export format)
    uint4 zero_fp16_seq;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.x) : "r"(zero_fp16.x), "r"(zero_fp16.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.y) : "r"(zero_fp16.x), "r"(zero_fp16.z));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(zero_fp16_seq.z) : "r"(zero_fp16.y), "r"(zero_fp16.w));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(zero_fp16_seq.w) : "r"(zero_fp16.y), "r"(zero_fp16.w));
    zero_fp16 = zero_fp16_seq;

    const uint4    scale_fp16 = *(uint4*)(scales_ptr2);
    const uint32_t B_packed   = *(uint32_t*)B_ptr2;
    uint4          B_deq      = dequantize_s4_to_fp16x2(B_packed);

    // Reorder interleaved AWQ qweight nibbles to sequential layout (export format)
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
// LAUNCHER: awq_gemm  (with AUTO SPLIT-K, OPT#3)
// ======================================================================================
// in_feats:         [M, IC]      fp16
// kernel:           [IC, OC/8]   int32 (packed int4)
// scaling_factors:  [IC/G, OC]   fp16
// zeros:            [IC/G, OC/8] int32 (packed int4)
// split_k_iters:    K-split factor. If 0 -> auto-adaptive based on grid size.
//
// Returns: [M, OC] fp16
//   - split_k == 1: direct fp16 output
//   - split_k > 1:  fp32 partial sums + custom reduction kernel
// ======================================================================================
torch::Tensor awq_gemm(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int64_t       split_k_iters
) {
    const int M  = _in_feats.size(0);
    const int IC = _in_feats.size(1);
    const int OC = _kernel.size(1) * 8;
    const int G  = IC / _scaling_factors.size(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel   = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto scales   = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros    = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    // ==========================================================================
    // Shape validation
    // ==========================================================================
    TORCH_CHECK(OC % 64 == 0,  "OC must be a multiple of cta_N = 64");
    TORCH_CHECK(OC % 8 == 0,   "OC must be a multiple of pack_num = 8");
    TORCH_CHECK(G % 32 == 0,   "Group size must be a multiple of 32");

    // ==========================================================================
    // Tile configuration
    // ==========================================================================
    const bool use_n128   = (OC % 128 == 0);
    const int j_tiles_128 = OC / 128;
    const int j_tiles_64  = OC / 64;
    const int j_tiles     = use_n128 ? j_tiles_128 : j_tiles_64;
    dim3 threads_per_block(32, 2);

    int effective_split_k = static_cast<int>(split_k_iters);
    if (split_k_iters == 0) {
        const int m_tiles = (M + 15) / 16;
        const int base_blocks = m_tiles * j_tiles;
        constexpr int target_sms = 80;  // V100-SXM2 has 80 SMs
        if (base_blocks < target_sms) {
            effective_split_k = (target_sms + base_blocks - 1) / base_blocks;
            effective_split_k = std::min(effective_split_k, 16);
        } else {
            effective_split_k = 1;
        }
    }

    // ==========================================================================
    // Dispatch: fast path (split_k=1) or split-k path
    // ==========================================================================
    if (effective_split_k == 1) {
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat16)
                           .device(_in_feats.device());
        at::Tensor _out_feats = torch::empty({M, OC}, options);
        auto out_feats = _out_feats.data_ptr<at::Half>();

        dim3 num_blocks((M + 16 - 1) / 16 * j_tiles);

        if (use_n128) {
            vllm::awq::gemm_forward_4bit_cuda_m16n16k16<128, false>
                <<<num_blocks, threads_per_block, 0, stream>>>(
                    G, 1, in_feats, kernel, scales, zeros,
                    M, IC, OC, out_feats);
        } else {
            vllm::awq::gemm_forward_4bit_cuda_m16n16k16<64, false>
                <<<num_blocks, threads_per_block, 0, stream>>>(
                    G, 1, in_feats, kernel, scales, zeros,
                    M, IC, OC, out_feats);
        }
        return _out_feats;
    } else {
        auto options_fp32 = torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(_in_feats.device());
        at::Tensor _split_feats = torch::empty({effective_split_k, M, OC}, options_fp32);
        auto split_feats = _split_feats.data_ptr<float>();

        dim3 num_blocks((M + 16 - 1) / 16 * j_tiles * effective_split_k);

        if (use_n128) {
            vllm::awq::gemm_forward_4bit_cuda_m16n16k16<128, true>
                <<<num_blocks, threads_per_block, 0, stream>>>(
                    G, effective_split_k, in_feats, kernel, scales, zeros,
                    M, IC, OC, split_feats);
        } else {
            vllm::awq::gemm_forward_4bit_cuda_m16n16k16<64, true>
                <<<num_blocks, threads_per_block, 0, stream>>>(
                    G, effective_split_k, in_feats, kernel, scales, zeros,
                    M, IC, OC, split_feats);
        }

        // Custom fast reduction: sum fp32 partials and cast to fp16 in one pass
        auto options_fp16 = torch::TensorOptions()
                                .dtype(torch::kFloat16)
                                .device(_in_feats.device());
        at::Tensor _out_feats = torch::empty({M, OC}, options_fp16);
        auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

        dim3 reduce_threads(16, 16);
        dim3 reduce_blocks((OC + 15) / 16, (M + 15) / 16);
        vllm::awq::split_k_reduce_kernel
            <<<reduce_blocks, reduce_threads, 0, stream>>>(
                split_feats, out_feats, M, OC, effective_split_k);

        return _out_feats;
    }
}