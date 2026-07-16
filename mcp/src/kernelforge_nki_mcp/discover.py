"""Repo walker — finds candidate kernels for conversion.

Three heuristics:
  1. `@triton.jit` decorated functions.
  2. `nn.Module.forward` methods that delegate to a hand-written CUDA custom
     kernel (a `load_inline` / `cpp_extension` op, or a module whose file
     carries CUDA C such as `__global__`). This is the headline migration case:
     an NVIDIA-only kernel that must be rewritten for Trainium.
  3. `nn.Module.forward` methods whose body matches a hot-op signature
     (matmul, softmax, layernorm, attention, RMSNorm, conv).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

HOT_OP_PATTERNS = (
    re.compile(r"\btorch\.matmul\b"),
    re.compile(r"\b@\b"),
    re.compile(r"\bF\.softmax\b"),
    re.compile(r"\bF\.layer_norm\b"),
    re.compile(r"\bF\.scaled_dot_product_attention\b"),
    re.compile(r"\bnn\.RMSNorm\b"),
    re.compile(r"\bF\.conv2d\b"),
)

# Markers that a file ships a hand-written CUDA custom kernel. Any of these in
# the file's source flags its module-forwards as CUDA-backed migration targets.
CUDA_FILE_PATTERNS = (
    re.compile(r"__global__"),                          # CUDA C kernel
    re.compile(r"\bload_inline\b"),                     # torch.utils.cpp_extension
    re.compile(r"\bcpp_extension\b"),
    re.compile(r"\bcuda_sources\b"),
    re.compile(r"""["']\s*(?:[^"']*/)?[\w-]+\.cu["']"""),  # a referenced .cu file
)

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "build", "dist"}


@dataclass
class Candidate:
    file: str
    line: int
    function: str
    signature: str
    kind: str  # "triton" | "cuda" | "module_forward"
    hot: bool
    already_nki: bool
    snippet: str = ""


@dataclass
class DiscoveryResult:
    repo_root: str
    candidates: list[Candidate] = field(default_factory=list)


def _iter_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _has_decorator(node: ast.FunctionDef, name: str) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Call):
            f = dec.func
            if isinstance(f, ast.Attribute) and f.attr == name:
                return True
            if isinstance(f, ast.Name) and f.id == name:
                return True
    return False


def _decorator_chain(node: ast.FunctionDef) -> str:
    parts = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute):
            parts.append(f"@{ast.unparse(dec)}")
        elif isinstance(dec, ast.Name):
            parts.append(f"@{dec.id}")
        elif isinstance(dec, ast.Call):
            parts.append(f"@{ast.unparse(dec.func)}")
    return ",".join(parts)


def _signature(node: ast.FunctionDef) -> str:
    return f"{node.name}({', '.join(a.arg for a in node.args.args)})"


def _body_is_hot(node: ast.FunctionDef, source: str) -> bool:
    src_segment = ast.get_source_segment(source, node) or ""
    return any(p.search(src_segment) for p in HOT_OP_PATTERNS)


def _file_has_cuda_kernel(source: str) -> bool:
    """True if the file ships a hand-written CUDA custom kernel."""
    return any(p.search(source) for p in CUDA_FILE_PATTERNS)


def _is_module_forward(node: ast.FunctionDef, parent_class: ast.ClassDef | None) -> bool:
    if node.name != "forward":
        return False
    if parent_class is None:
        return False
    for base in parent_class.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
    return False


def discover(repo_root: str) -> DiscoveryResult:
    root = Path(repo_root).resolve()
    result = DiscoveryResult(repo_root=str(root))
    if not root.exists():
        return result

    for path in _iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue

        # Walk top-level functions and class methods.
        for outer in ast.walk(tree):
            if isinstance(outer, ast.ClassDef):
                for item in outer.body:
                    if isinstance(item, ast.FunctionDef):
                        _maybe_emit(item, outer, path, source, result)
            elif isinstance(outer, ast.FunctionDef):
                _maybe_emit(outer, None, path, source, result)

    return result


def _maybe_emit(
    node: ast.FunctionDef,
    parent: ast.ClassDef | None,
    path: Path,
    source: str,
    result: DiscoveryResult,
) -> None:
    if _has_decorator(node, "jit"):
        chain = _decorator_chain(node)
        if "triton" in chain:
            kind = "triton"
        elif "nki" in chain:
            kind = "nki"
        else:
            return
    elif _is_module_forward(node, parent):
        # A module forward in a file that ships a CUDA custom kernel is the
        # headline migration target (NVIDIA-only code that must move to
        # Trainium), whether or not its body matches a torch hot-op pattern.
        if _file_has_cuda_kernel(source):
            kind = "cuda"
        elif _body_is_hot(node, source):
            kind = "module_forward"
        else:
            return
    else:
        return

    result.candidates.append(
        Candidate(
            file=str(path),
            line=node.lineno,
            function=node.name,
            signature=_signature(node),
            kind=kind if kind in ("triton", "cuda") else "module_forward",
            hot=True,
            already_nki=(kind == "nki"),
            snippet=(ast.get_source_segment(source, node) or "")[:1500],
        )
    )
