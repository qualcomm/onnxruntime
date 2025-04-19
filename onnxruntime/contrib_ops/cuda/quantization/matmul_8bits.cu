// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = WARP_SHFL((unsigned)p.asInt, 0);
  return p.value;
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __ushort2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  // Extract 8 uint8_t values from the 64-bit input.
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // Convert pairs to half2 (0,4 1,5 2,6 3,7 interleaved order).
  half2 element04 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[4]));
  half2 element15 = __halves2half2(__ushort2half_rn(q[1]), __ushort2half_rn(q[5]));
  half2 element26 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[6]));
  half2 element37 = __halves2half2(__ushort2half_rn(q[3]), __ushort2half_rn(q[7]));

  half2 v0 = element04 * scale_half2 + zp_adjust2;
  half2 v1 = element15 * scale_half2 + zp_adjust2;
  half2 v2 = element26 * scale_half2 + zp_adjust2;
  half2 v3 = element37 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xFF) * scale + zp_adjust;
  float v1 = float((values_quant >> 8) & 0xFF) * scale + zp_adjust;
  float v2 = float((values_quant >> 16) & 0xFF) * scale + zp_adjust;
  float v3 = float((values_quant >> 24) & 0xFF) * scale + zp_adjust;
  float v4 = float((values_quant >> 32) & 0xFF) * scale + zp_adjust;
  float v5 = float((values_quant >> 40) & 0xFF) * scale + zp_adjust;
  float v6 = float((values_quant >> 48) & 0xFF) * scale + zp_adjust;
  float v7 = float((values_quant >> 56) & 0xFF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = GPU_WARP_SIZE;

// kernel for 8bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 8bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat8bKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
    const int b_zp_k = blocks_per_K;
    int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
      b_zp_vec[i] = zero_points[zp_offset + i];
    }
    b_zp_vec += warp_id * b_zp_k;
  }
  __syncthreads();

  a_data += m_id * k + (lane_id << 3);

  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                          \
  do {                                                                                        \
    constexpr int kUnroll = unroll_size;                                                      \
    constexpr int kUnrollMask = 0xffffffff & (~(kUnroll * k_per_iter - 1));                   \
    for (; k_id < (k & kUnrollMask); k_id += kUnroll * k_per_iter) {                          \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                   \
        uint64_t value = *(reinterpret_cast<const uint64_t*>(b_data_quant + k_per_iter * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                        \
        uint8_t zp = 8;                                                                       \
        if constexpr (has_zero_point) {                                                       \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                              \
        }                                                                                     \
        AccumulateEightElements8b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);    \
      }                                                                                       \
      b_data_quant += k_per_iter * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                          \
    }                                                                                         \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint64_t value = *(reinterpret_cast<const uint64_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements8b(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  // warp reduction
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}  // namespace cuda

template <class T>
bool TryMatMul8Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                        (zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

#define MatMulFloat8bKernelDispatch(block_size)                                              \
  if (nullptr != zero_points) {                                                              \
    MatMulFloat8bKernel<T, block_size, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);      \
  } else {                                                                                   \
    MatMulFloat8bKernel<T, block_size, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);      \
  }

  if (16 == block_size) {
    MatMulFloat8bKernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloat8bKernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloat8bKernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloat8bKernelDispatch(128);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

#undef MatMulFloat8bKernelDispatch

  return true;
}

template bool TryMatMul8Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul8Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
