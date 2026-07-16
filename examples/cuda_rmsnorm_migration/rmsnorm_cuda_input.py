"""Example input: a hand-written CUDA custom kernel for RMSNorm.

This is the exact shape of migration this agent is built for: an open-source
repo ships a hand-tuned CUDA kernel (here, a fused RMSNorm compiled with
``torch.utils.cpp_extension.load_inline``) that no generic porting tool will
safely rewrite for Trainium — a numerically wrong rewrite silently corrupts
model outputs, and a slow one defeats the point of migrating.

RMSNorm is the normalization used in LLaMA / Mistral / Qwen and most modern
LLMs, and is almost always shipped as a custom CUDA kernel for throughput.
Representative of e.g. the reference kernels in HuggingFace/`apex`-style repos.

The custom op below (a) only runs on an NVIDIA GPU (``__global__`` CUDA C), and
(b) carries a plain-PyTorch reference (``RMSNormReference``) that defines the
exact numerics the migrated NKI kernel must match. The agent discovers the hot
``forward``, generates an ``@nki.jit`` kernel, then compiles / verifies /
profiles it against this reference on a real Trainium device.
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

# ── The custom CUDA kernel (NVIDIA-only) ──────────────────────────────────────
# One block per row; each block reduces sum-of-squares across the hidden dim,
# then normalizes and applies the learned weight. This is the artifact that
# pins the migration to NVIDIA hardware.
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,       // [rows, hidden]
    const scalar_t* __restrict__ weight,  // [hidden]
    scalar_t* __restrict__ out,           // [rows, hidden]
    const int hidden,
    const float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* x_row = x + row * hidden;
  scalar_t* out_row = out + row * hidden;

  // Block-wide reduction of sum of squares.
  extern __shared__ float sdata[];
  float local = 0.f;
  for (int i = tid; i < hidden; i += blockDim.x) {
    const float v = static_cast<float>(x_row[i]);
    local += v * v;
  }
  sdata[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sdata[0] / hidden + eps);

  for (int i = tid; i < hidden; i += blockDim.x) {
    const float v = static_cast<float>(x_row[i]) * inv_rms;
    out_row[i] = static_cast<scalar_t>(v) * weight[i];
  }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const auto rows = x.size(0);
  const auto hidden = x.size(1);
  auto out = torch::empty_like(x);
  const int threads = 256;
  const int shared = threads * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rmsnorm_cuda", ([&] {
    rmsnorm_kernel<scalar_t><<<rows, threads, shared>>>(
        x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(), hidden, static_cast<float>(eps));
  }));
  return out;
}
"""

_CPP_DECL = "torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);"


def _load():
    # Compiles only where nvcc + an NVIDIA GPU are present. On Trainium/CPU this
    # raises — which is exactly the wall a migration hits, and why the agent
    # regenerates the kernel from the reference below rather than this source.
    return load_inline(
        name="rmsnorm_cuda_ext",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=["rmsnorm_cuda"],
        verbose=False,
    )


class RMSNormCUDA(torch.nn.Module):
    """RMSNorm backed by the custom CUDA kernel above (NVIDIA-only)."""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))
        self.eps = eps
        self._ext = _load()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ext.rmsnorm_cuda(x, self.weight, self.eps)


class RMSNormReference(torch.nn.Module):
    """Plain-PyTorch reference — the numerics the migrated NKI kernel must match.

    Device-agnostic on purpose: this is the ground truth the reward server runs
    on real Trainium silicon inside ``nki_verify`` (atol = rtol = 1e-3).
    """

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


def get_inputs():
    """Representative shape: (batch*seq, hidden) = (4096, 4096), fp32."""
    return [torch.randn(4096, 4096, dtype=torch.float32)]
