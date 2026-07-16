"""PyTorch reference for the row-wise softmax.

Used by the verify pipeline. The reward server's verify subprocess imports this
file and calls `reference(x)` with a torch tensor matching the kernel's input
spec. Output is compared with numpy.allclose.
"""

from __future__ import annotations

import torch


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
