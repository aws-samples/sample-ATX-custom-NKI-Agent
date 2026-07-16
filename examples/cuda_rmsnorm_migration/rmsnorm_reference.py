"""Ground-truth reference for the RMSNorm migration.

This is the device-agnostic PyTorch definition the migrated NKI kernel must
match. The reward server runs this on real Trainium silicon inside
``nki_verify`` and compares element-wise against the generated kernel
(``np.allclose``, atol = rtol = 1e-3). It is intentionally the *reference*, not
the CUDA op in ``rmsnorm_cuda_input.py`` — the whole point of the migration is
that the CUDA kernel does not run on Trainium, so numerical equivalence is
defined against plain PyTorch.
"""

from __future__ import annotations

import torch


class RMSNormReference(torch.nn.Module):
    def __init__(self, hidden: int = 4096, eps: float = 1e-6):
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
