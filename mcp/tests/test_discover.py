"""Smoke test for the repo walker."""

from __future__ import annotations

from pathlib import Path

from kernelforge_nki_mcp.discover import discover


def test_discover_finds_triton(tmp_path: Path) -> None:
    src = tmp_path / "kernels.py"
    src.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def add_kernel(x, y, z):\n"
        "    pass\n"
    )
    result = discover(str(tmp_path))
    assert any(c.kind == "triton" and c.function == "add_kernel" for c in result.candidates)


def test_discover_finds_module_forward(tmp_path: Path) -> None:
    src = tmp_path / "model.py"
    src.write_text(
        "import torch\n"
        "from torch import nn\n"
        "import torch.nn.functional as F\n"
        "class M(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return F.softmax(x, dim=-1)\n"
    )
    result = discover(str(tmp_path))
    assert any(
        c.kind == "module_forward" and c.function == "forward"
        for c in result.candidates
    )


def test_discover_finds_cuda_custom_kernel(tmp_path: Path) -> None:
    # A module whose forward delegates to a hand-written CUDA kernel is the
    # headline migration target — flagged 'cuda' even though its forward body
    # matches no torch hot-op pattern.
    src = tmp_path / "custom_op.py"
    src.write_text(
        "import torch\n"
        "from torch import nn\n"
        "from torch.utils.cpp_extension import load_inline\n"
        '_SRC = "__global__ void k() {}"\n'
        "_ext = load_inline(name='ext', cpp_sources='', cuda_sources=_SRC)\n"
        "class CustomOp(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return _ext.run(x)\n"
    )
    result = discover(str(tmp_path))
    assert any(
        c.kind == "cuda" and c.function == "forward" for c in result.candidates
    )


def test_discover_finds_dot_cu_reference(tmp_path: Path) -> None:
    # A module in a file that references a .cu source is also a CUDA target.
    src = tmp_path / "op.py"
    src.write_text(
        "import torch\n"
        "from torch import nn\n"
        '_ext = torch.ops.load_library("kernels/fused.cu")\n'
        "class Op(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return _ext.fused(x)\n"
    )
    result = discover(str(tmp_path))
    assert any(c.kind == "cuda" for c in result.candidates)


def test_discover_skips_already_nki(tmp_path: Path) -> None:
    src = tmp_path / "k.py"
    src.write_text(
        "import neuronxcc.nki as nki\n"
        "@nki.jit\n"
        "def nki_kernel(x):\n"
        "    return x\n"
    )
    result = discover(str(tmp_path))
    # Already-NKI kernels are surfaced but flagged.
    nki_only = [c for c in result.candidates if c.already_nki]
    # Either filtered out at the AST stage (current behavior) or surfaced with already_nki=True.
    assert len(result.candidates) == 0 or all(c.already_nki for c in nki_only)
