// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "../cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/w8a8/fp8/common.cuh"

namespace {

constexpr int kWaveSize = 32;
constexpr int kWavesPerWG = 16;
constexpr int kThreadsPerWG = kWaveSize * kWavesPerWG;
constexpr int kAChunk = 16;
constexpr int kYTile = 2;
constexpr int kFixedK = 4096;
constexpr int kGroupK = 128;
constexpr int kFixedKGroups = kFixedK / kGroupK;
constexpr int kMaxTokens = 6;
constexpr std::uintptr_t kVecLoadBytes = 16;

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
typedef short __attribute__((ext_vector_type(2))) bf16x2_t;

struct bf16_16_t {
  __nv_bfloat162 p0;
  __nv_bfloat162 p1;
  __nv_bfloat162 p2;
  __nv_bfloat162 p3;
  __nv_bfloat162 p4;
  __nv_bfloat162 p5;
  __nv_bfloat162 p6;
  __nv_bfloat162 p7;
};

bool on_gfx1151() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  return std::string(props->gcnArchName).find("gfx1151") == 0;
}

#if defined(__GFX11__) || defined(__GFX12__)
__device__ __forceinline__ float dot2_bf16(__nv_bfloat162 lhs,
                                           __nv_bfloat162 rhs, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      *reinterpret_cast<const bf16x2_t*>(&lhs),
      *reinterpret_cast<const bf16x2_t*>(&rhs), acc, false);
}

__device__ __forceinline__ float dot16_bf16(const bf16_16_t& lhs,
                                            const bf16_16_t& rhs) {
  float acc = 0.0f;
  acc = dot2_bf16(lhs.p0, rhs.p0, acc);
  acc = dot2_bf16(lhs.p1, rhs.p1, acc);
  acc = dot2_bf16(lhs.p2, rhs.p2, acc);
  acc = dot2_bf16(lhs.p3, rhs.p3, acc);
  acc = dot2_bf16(lhs.p4, rhs.p4, acc);
  acc = dot2_bf16(lhs.p5, rhs.p5, acc);
  acc = dot2_bf16(lhs.p6, rhs.p6, acc);
  acc = dot2_bf16(lhs.p7, rhs.p7, acc);
  return acc;
}

__device__ __forceinline__ float wave32_reduce_sum(float x) {
  x += __builtin_amdgcn_mov_dpp(x, 0x118, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x114, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x112, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x111, 0xf, 0xf, 1);
  x += __shfl_xor(x, 16);
  return x;
}

__device__ __forceinline__ float subgroup_broadcast8(float x, int lane) {
  return __shfl(x, lane & ~7, kWaveSize);
}

__device__ __forceinline__ bf16_16_t fp8x16_to_bf16(uint4 packed) {
  union packed16_t {
    uint4 u4;
    uint16_t u16[8];
  } tmp;
  tmp.u4 = packed;

  bf16_16_t out;
  out.p0 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[0], 1.0f);
  out.p1 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[1], 1.0f);
  out.p2 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[2], 1.0f);
  out.p3 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[3], 1.0f);
  out.p4 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[4], 1.0f);
  out.p5 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[5], 1.0f);
  out.p6 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[6], 1.0f);
  out.p7 = vllm::fp8::scaled_vec_conversion<__nv_bfloat162, uint16_t>(
      tmp.u16[7], 1.0f);
  return out;
}

template <int TOKENS>
__global__
__launch_bounds__(kThreadsPerWG) void wvSplitKQBlockScaleGfx1151Kernel(
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale, __hip_bfloat16* __restrict__ out,
    int N, int cu_count) {
  static_assert(TOKENS >= 1 && TOKENS <= kMaxTokens);

  __shared__ uint4 sA[(TOKENS * kFixedK) / kAChunk];

  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int linear_tid = wave * kWaveSize + lane;

  for (int idx = linear_tid * kAChunk; idx < TOKENS * kFixedK;
       idx += kThreadsPerWG * kAChunk) {
    sA[idx / kAChunk] = *reinterpret_cast<const uint4*>(activation + idx);
  }
  __syncthreads();

  int out_n = (blockIdx.x * kWavesPerWG + wave) * kYTile;
  while (out_n < N) {
    float sum[TOKENS][kYTile] = {};

    for (int k_base = 0; k_base < kFixedK; k_base += kWaveSize * kAChunk) {
      const int logical_k = k_base + lane * kAChunk;
      const int k_group = logical_k / kGroupK;

      bf16_16_t weight_frag[kYTile];
      bool active_y[kYTile];
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        const int n = out_n + y;
        active_y[y] = n < N;
        if (active_y[y]) {
          const uint8_t* w = weight + n * kFixedK + logical_k;
          weight_frag[y] = fp8x16_to_bf16(*reinterpret_cast<const uint4*>(w));
        } else {
          weight_frag[y] = {};
        }
      }

      float ws[kYTile] = {};
      if ((lane & 7) == 0) {
  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          if (active_y[y]) {
            const int n_group = (out_n + y) / kGroupK;
            ws[y] = weight_scale[n_group * kFixedKGroups + k_group];
          }
        }
      }
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        ws[y] = subgroup_broadcast8(ws[y], lane);
      }

  #pragma unroll
      for (int token = 0; token < TOKENS; ++token) {
        const uint4 packed_a = sA[(token * kFixedK + logical_k) / kAChunk];
        const bf16_16_t act_frag = fp8x16_to_bf16(packed_a);

        float as = 0.0f;
        if ((lane & 7) == 0) {
          as = activation_scale[token * kFixedKGroups + k_group];
        }
        as = subgroup_broadcast8(as, lane);

  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          if (active_y[y]) {
            sum[token][y] += dot16_bf16(act_frag, weight_frag[y]) * as * ws[y];
          }
        }
      }
    }

  #pragma unroll
    for (int token = 0; token < TOKENS; ++token) {
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        sum[token][y] = wave32_reduce_sum(sum[token][y]);
      }
    }

    if (lane == kWaveSize - 1) {
  #pragma unroll
      for (int token = 0; token < TOKENS; ++token) {
  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          const int n = out_n + y;
          if (n < N) {
            out[token * N + n] = __float2bfloat16(sum[token][y]);
          }
        }
      }
    }

    out_n += cu_count * kWavesPerWG * kYTile;
  }
}
#else
template <int TOKENS>
__global__ void wvSplitKQBlockScaleGfx1151Kernel(const uint8_t*, const uint8_t*,
                                                 const float*, const float*,
                                                 __hip_bfloat16*, int, int) {}
#endif

template <int TOKENS, typename fp8_t>
void launch_wvsplitk_blockscale_gfx1151(const at::Tensor& weight,
                                        const at::Tensor& activation,
                                        const at::Tensor& activation_scale,
                                        const at::Tensor& weight_scale,
                                        at::Tensor& out, int64_t cu_count) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(activation));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto* weight_ptr = reinterpret_cast<const uint8_t*>(weight.data_ptr<fp8_t>());
  auto* act_ptr =
      reinterpret_cast<const uint8_t*>(activation.data_ptr<fp8_t>());
  auto* out_ptr =
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<c10::BFloat16>());

  dim3 grid(static_cast<unsigned int>(cu_count));
  dim3 block(kWaveSize, kWavesPerWG);
  wvSplitKQBlockScaleGfx1151Kernel<TOKENS><<<grid, block, 0, stream>>>(
      weight_ptr, act_ptr, activation_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(), out_ptr, static_cast<int>(weight.size(0)),
      static_cast<int>(cu_count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void wvSplitKQBlockScale(const at::Tensor& weight, const at::Tensor& activation,
                         const at::Tensor& activation_scale,
                         const at::Tensor& weight_scale, at::Tensor& out,
                         const int64_t CuCount) {
  TORCH_CHECK(weight.is_cuda() && activation.is_cuda() &&
                  activation_scale.is_cuda() && weight_scale.is_cuda() &&
                  out.is_cuda(),
              "wvSplitKQBlockScale expects HIP tensors");
  TORCH_CHECK(weight.device() == activation.device() &&
                  weight.device() == activation_scale.device() &&
                  weight.device() == weight_scale.device() &&
                  weight.device() == out.device(),
              "wvSplitKQBlockScale tensors must share a device");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(activation));
  TORCH_CHECK(on_gfx1151(), "wvSplitKQBlockScale is supported on gfx1151 only");
  const c10::ScalarType fp8_type = c10::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(
      weight.scalar_type() == fp8_type && activation.scalar_type() == fp8_type,
      "weight and activation must use the native ROCm fp8 dtype");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "block scales must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16,
              "wvSplitKQBlockScale output must be bfloat16");

  TORCH_CHECK(weight.dim() == 2 && activation.dim() == 2 &&
                  activation_scale.dim() == 2 && weight_scale.dim() == 2 &&
                  out.dim() == 2,
              "wvSplitKQBlockScale expects rank-2 tensors");
  TORCH_CHECK(weight.is_contiguous(),
              "weight must be contiguous logical [N,K] layout");
  TORCH_CHECK(activation.is_contiguous(),
              "activation must be contiguous [M,K]");
  TORCH_CHECK(activation_scale.is_contiguous(),
              "activation_scale must be contiguous [M,K/128]");
  TORCH_CHECK(weight_scale.is_contiguous(),
              "weight_scale must be contiguous [N/128,K/128]");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous [M,N]");

  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxTokens, "M must be in [1,6], got ", M);
  TORCH_CHECK(K == kFixedK && weight.size(1) == kFixedK,
              "logical K must be 4096");
  TORCH_CHECK(N == 1536 || N == 4096, "logical N must be 1536 or 4096, got ",
              N);
  TORCH_CHECK(activation_scale.size(0) == M &&
                  activation_scale.size(1) == kFixedKGroups,
              "activation_scale must be [M,32], got ",
              activation_scale.sizes());
  TORCH_CHECK(weight_scale.size(0) == N / kGroupK &&
                  weight_scale.size(1) == kFixedKGroups,
              "weight_scale must be [N/128,32], got ", weight_scale.sizes());
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out must be [M,N], got ",
              out.sizes());
  TORCH_CHECK(
      CuCount > 0 &&
          CuCount <=
              at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
      "CuCount must be in [1, device CU count]");

  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % kVecLoadBytes ==
              0 &&
          reinterpret_cast<std::uintptr_t>(activation.data_ptr()) %
                  kVecLoadBytes ==
              0,
      "fp8 inputs must be 16-byte aligned");

  VLLM_DISPATCH_FP8_TYPES(activation.scalar_type(), "wvSplitKQBlockScale", [&] {
    switch (M) {
      case 1:
        launch_wvsplitk_blockscale_gfx1151<1, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 2:
        launch_wvsplitk_blockscale_gfx1151<2, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 3:
        launch_wvsplitk_blockscale_gfx1151<3, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 4:
        launch_wvsplitk_blockscale_gfx1151<4, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 5:
        launch_wvsplitk_blockscale_gfx1151<5, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 6:
        launch_wvsplitk_blockscale_gfx1151<6, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      default:
        TORCH_CHECK(false, "unsupported M=", M);
    }
  });
}
