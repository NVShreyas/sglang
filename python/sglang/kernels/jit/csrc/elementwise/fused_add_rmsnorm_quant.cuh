// Fused Add-residual + RMSNorm + 1x128 fp8 block-quant producer (SM90). Runtime H.
//
// Replaces the stock 2-kernel producer (fused_add_rmsnorm then per_token_group_quant_fp8)
// that runs before fp8 block-scaled linears in deepseek_v2-family models. One CTA per row;
// NT = H/VPT threads cover the whole row in a single pass (one barrier). VPT is the only
// template param so the per-thread register arrays stay compile-time sized. Memory-bound, so
// the host dispatch (wrapper) picks VPT by H alone: VPT=8 for H<=8192, VPT=16 for H<=16384.
// Requires H % (32*VPT) == 0 (H%256 for VPT8, H%512 for VPT16; both subsume H%128 for the
// 1x128 quant blocks). Input/residual/weight/normed are bf16; the wrapper guards the dtype.
//
// Emits: fp8 e4m3 [M,H]; column-major TMA-aligned scale in a (H/128, Mpad) buffer with
// Mpad = round_up(M,4) (the MN-major layout the DeepGEMM / figemm pre-quant path expects);
// residual_out bf16 (hidden+residual, pre-norm); normed_out bf16 (so bf16 consumers keep an
// input while the fp8 linears consume x_fp8/x_scale). NOT in-place: inputs are untouched.
// The JIT host framework (SymbolicSize/SymbolicDevice/TensorMatcher/RuntimeCheck/LaunchKernel)
// lives in namespace sglang::host; bf16_t is in type.cuh. Match the reference kernel's include set;
// raw CUDA headers last so the sgl_kernel type headers parse first.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

#define KF_FMAX 448.0f

template <int VPT>
__global__ void __launch_bounds__(1024) fused_add_rmsnorm_quant_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ x_fp8,
    float* __restrict__ x_scale,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ normed_out,
    int H,
    float eps) {
  constexpr int LPB = 128 / VPT;  // threads forming one 1x128 quant block
  constexpr int NV = VPT / 8;     // uint4 (8 bf16) chunks per thread
  const int NT = blockDim.x;      // == H/VPT
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int col = tid * VPT;
  const size_t base = (size_t)row * H + col;
  const unsigned M = gridDim.x;

  uint4 h4[NV], r4[NV], w4[NV];
#pragma unroll
  for (int v = 0; v < NV; v++) {
    h4[v] = *reinterpret_cast<const uint4*>(hidden + base + v * 8);
    r4[v] = *reinterpret_cast<const uint4*>(residual + base + v * 8);
    w4[v] = *reinterpret_cast<const uint4*>(weight + col + v * 8);
  }

  float2 res[NV * 4];
  float sumsq = 0.f;
#pragma unroll
  for (int v = 0; v < NV; v++) {
    const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&h4[v]);
    const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&r4[v]);
    __nv_bfloat162 rb[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 hf = __bfloat1622float2(hp[k]);
      float2 rf = __bfloat1622float2(rp[k]);
      float2 val = make_float2(hf.x + rf.x, hf.y + rf.y);
      res[v * 4 + k] = val;
      sumsq += val.x * val.x + val.y * val.y;
      rb[k] = __float22bfloat162_rn(val);
    }
    *reinterpret_cast<uint4*>(residual_out + base + v * 8) = *reinterpret_cast<const uint4*>(rb);
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sumsq += __shfl_down_sync(0xffffffffu, sumsq, o);
  __shared__ float ssm[32];  // max NT/32 (NT <= 1024)
  const int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) ssm[warp] = sumsq;
  __syncthreads();
  float total = 0.f;
  const int nwarps = NT >> 5;
  for (int i = 0; i < nwarps; i++) total += ssm[i];
  const float rms = rsqrtf(total / (float)H + eps);

  float2 normed[NV * 4];
  float lamax = 0.f;
#pragma unroll
  for (int v = 0; v < NV; v++) {
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w4[v]);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int idx = v * 4 + k;
      float2 wf = __bfloat1622float2(wp[k]);
      float2 n = make_float2(res[idx].x * rms * wf.x, res[idx].y * rms * wf.y);
      // round to bf16 BEFORE amax/quant so fp8 is quantized from the bf16 normed (matches figemm).
      float2 nbf = __bfloat1622float2(__float22bfloat162_rn(n));
      normed[idx] = nbf;
      lamax = fmaxf(lamax, fmaxf(fabsf(nbf.x), fabsf(nbf.y)));
    }
  }
#pragma unroll
  for (int o = LPB / 2; o >= 1; o >>= 1) lamax = fmaxf(lamax, __shfl_xor_sync(0xffffffffu, lamax, o));

  const float scale = fmaxf(lamax, 1e-4f) / KF_FMAX;
  if ((tid % LPB) == 0) {
    const unsigned Mpad = (M + 3u) & ~3u;  // round_up(M,4), TMA-aligned column-major
    x_scale[(size_t)(tid / LPB) * Mpad + row] = scale;
  }
  const float inv = KF_FMAX / fmaxf(lamax, 1e-4f);
#pragma unroll
  for (int v = 0; v < NV; v++) {
    __nv_fp8x2_e4m3 q[4];
    __nv_bfloat162 nb[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 nrm = normed[v * 4 + k];
      float2 s = make_float2(nrm.x * inv, nrm.y * inv);
      q[k] = __nv_fp8x2_e4m3(s);
      nb[k] = __float22bfloat162_rn(nrm);
    }
    *reinterpret_cast<uint2*>(x_fp8 + base + v * 8) = *reinterpret_cast<const uint2*>(q);
    *reinterpret_cast<uint4*>(normed_out + base + v * 8) = *reinterpret_cast<const uint4*>(nb);
  }
}

// Host launcher. VPT is fixed at compile time; NT = H/VPT is the runtime block size. The Python
// wrapper guarantees bf16 inputs, H % (32*VPT) == 0, and NT <= 1024 (so the reduction and the
// register arrays are valid), and allocates the four output tensors with the layouts above.
template <int VPT>
struct FusedAddRMSNormQuantKernel {
  static void
  run(tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView x_fp8,
      tvm::ffi::TensorView x_scale,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView normed_out,
      double eps) {
    using namespace sglang;        // JIT framework namespace (7c90840b)
    using namespace sglang::host;  // SymbolicSize / SymbolicDevice / TensorMatcher / RuntimeCheck / LaunchKernel
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({D, 1}).with_dtype<bf16_t>().with_device(device).verify(hidden);
    TensorMatcher({N, D}).with_strides({D, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(weight);

    const int H = static_cast<int>(D.unwrap());
    const uint M = static_cast<uint>(N.unwrap());
    const uint NT = static_cast<uint>(H / VPT);
    host::RuntimeCheck(H % (32 * VPT) == 0, "hidden_size ", H, " not a multiple of ", 32 * VPT);
    host::RuntimeCheck(NT <= 1024, "H/VPT ", NT, " exceeds 1024 threads");
    if (M == 0) return;

    auto kernel = fused_add_rmsnorm_quant_kernel<VPT>;
    host::LaunchKernel(M, NT, device.unwrap())
        .enable_pdl(false)(
            kernel,
            reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
            reinterpret_cast<float*>(x_scale.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr()),
            H,
            static_cast<float>(eps));
  }
};

}  // namespace
