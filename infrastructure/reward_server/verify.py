"""On-device verification: run the NKI kernel via @nki.baremetal and the
PyTorch reference module, compare element-wise with numpy.allclose.

Same subprocess isolation as compile.py — Neuron runtime crashes don't take
down the reward server.

Caller payload:
  kernel_source    str   — NKI kernel source (must define `nki_kernel`)
  reference_source str   — PyTorch reference (must define `reference(*args)`)
  input_specs      list  — [{"shape": [...], "dtype": "float32"|...}]
  atol             float — absolute tolerance (default 1e-3)
  rtol             float — relative tolerance (default 1e-3)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from reward_server.sandbox_env import build_subprocess_env

MAX_SOURCE_BYTES = 256 * 1024
_TMP_PATH_RE = re.compile(r"/tmp/[A-Za-z0-9_/.\-]+")  # nosec B108 — regex literal used to redact /tmp paths from log output, not an insecure tempfile call
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_DTYPE_RE = re.compile(r"^(float16|float32|bfloat16|int8|int16|int32)$")


def _sanitize(line: str) -> str:
    return _TMP_PATH_RE.sub("<tmp>", line)


def verify_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("kernel_source")
    reference = payload.get("reference_source")
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE_BYTES:
        return _err("kernel_source missing or too large")
    if not isinstance(reference, str) or len(reference.encode()) > MAX_SOURCE_BYTES:
        return _err("reference_source missing or too large")

    kernel_name = payload.get("kernel_name", "nki_kernel")
    if not _IDENT_RE.match(kernel_name):
        return _err("kernel_name must be a Python identifier")

    input_specs = payload.get("input_specs")
    if not isinstance(input_specs, list) or not input_specs:
        return _err("input_specs must be a non-empty list")
    for spec in input_specs:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("shape"), list)
            or not all(isinstance(d, int) and d > 0 for d in spec["shape"])
            or not _DTYPE_RE.match(str(spec.get("dtype", "")))
        ):
            return _err("each input_spec needs shape=list[int] and dtype=float*/int*")

    atol = float(payload.get("atol", 1e-3))
    rtol = float(payload.get("rtol", 1e-3))

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="nki-verify-") as tmpdir:
        kpath = Path(tmpdir) / f"{kernel_name}.py"
        rpath = Path(tmpdir) / "ref.py"
        runner = Path(tmpdir) / "runner.py"
        kpath.write_text(source, encoding="utf-8")
        rpath.write_text(reference, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")

        env = build_subprocess_env(
            {
                "NKI_KERNEL_FILE": str(kpath),
                "NKI_REF_FILE": str(rpath),
                "NKI_KERNEL_NAME": kernel_name,
                "NKI_INPUT_SPECS": json.dumps(input_specs),
                "NKI_ATOL": str(atol),
                "NKI_RTOL": str(rtol),
            }
        )

        try:
            r = subprocess.run(
                [sys.executable, str(runner)],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return _err("verify timeout (>600s)", elapsed=time.time() - start)

        # The runner emits a JSON line on stdout containing the metrics.
        result_line = ""
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("__VERIFY__"):
                result_line = ln[len("__VERIFY__"):]
        try:
            metrics = json.loads(result_line) if result_line else {}
        except json.JSONDecodeError:
            metrics = {}

        if r.returncode != 0:
            return {
                "correct": False,
                "max_abs_error": metrics.get("max_abs_error"),
                "max_rel_error": metrics.get("max_rel_error"),
                "mismatched_elements": metrics.get("mismatched_elements"),
                "total_elements": metrics.get("total_elements"),
                "atol": atol,
                "rtol": rtol,
                "verify_time_s": round(time.time() - start, 2),
                "error": _sanitize_block(r.stderr or "")[-1500:],
            }

        return {
            "correct": bool(metrics.get("correct", False)),
            "max_abs_error": metrics.get("max_abs_error"),
            "max_rel_error": metrics.get("max_rel_error"),
            "mismatched_elements": metrics.get("mismatched_elements"),
            "total_elements": metrics.get("total_elements"),
            "atol": atol,
            "rtol": rtol,
            "verify_time_s": round(time.time() - start, 2),
            "error": None,
        }


def _err(msg: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "correct": False,
        "max_abs_error": None,
        "max_rel_error": None,
        "mismatched_elements": None,
        "total_elements": None,
        "atol": None,
        "rtol": None,
        "verify_time_s": round(elapsed, 2),
        "error": msg,
    }


def _sanitize_block(s: str) -> str:
    return "\n".join(_sanitize(line) for line in s.splitlines())


_RUNNER = '''\
import importlib.util, json, os, sys, traceback
import numpy as np

kpath = os.environ["NKI_KERNEL_FILE"]
rpath = os.environ["NKI_REF_FILE"]
kname = os.environ["NKI_KERNEL_NAME"]
specs = json.loads(os.environ["NKI_INPUT_SPECS"])
atol  = float(os.environ["NKI_ATOL"])
rtol  = float(os.environ["NKI_RTOL"])

def _emit(metrics):
    sys.stdout.write("__VERIFY__" + json.dumps(metrics) + "\\n")
    sys.stdout.flush()

try:
    import torch
    import neuronxcc.nki as nki
except Exception as e:
    print(f"error: import failed: {e}", file=sys.stderr)
    sys.exit(2)

DTYPE = {
    "float16": (np.float16, torch.float16),
    "float32": (np.float32, torch.float32),
    "bfloat16": (np.float32, torch.bfloat16),
    "int8":    (np.int8, torch.int8),
    "int16":   (np.int16, torch.int16),
    "int32":   (np.int32, torch.int32),
}

rng = np.random.default_rng(42)
np_inputs, t_inputs = [], []
for s in specs:
    np_dtype, t_dtype = DTYPE[s["dtype"]]
    arr = rng.standard_normal(size=tuple(s["shape"])).astype(np_dtype)
    np_inputs.append(arr)
    t_inputs.append(torch.from_numpy(arr.astype(np.float32)).to(t_dtype))

try:
    spec = importlib.util.spec_from_file_location(kname, kpath)
    kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)
except Exception:
    print("error: kernel import failed", file=sys.stderr); traceback.print_exc(); sys.exit(3)

try:
    rspec = importlib.util.spec_from_file_location("ref", rpath)
    rmod = importlib.util.module_from_spec(rspec); rspec.loader.exec_module(rmod)
except Exception:
    print("error: reference import failed", file=sys.stderr); traceback.print_exc(); sys.exit(4)

if not hasattr(kmod, "nki_kernel"):
    print("error: source defines no nki_kernel function", file=sys.stderr); sys.exit(5)
if not hasattr(rmod, "reference"):
    print("error: reference defines no `reference(*args)` function", file=sys.stderr); sys.exit(6)

try:
    bm = nki.baremetal(kmod.nki_kernel)
    nki_out = bm(*np_inputs)
    nki_arr = np.asarray(nki_out)
except Exception:
    print("error: nki baremetal run failed", file=sys.stderr); traceback.print_exc(); sys.exit(7)

try:
    with torch.no_grad():
        ref_out = rmod.reference(*t_inputs)
    ref_arr = ref_out.to(torch.float32).cpu().numpy()
except Exception:
    print("error: reference run failed", file=sys.stderr); traceback.print_exc(); sys.exit(8)

if nki_arr.shape != ref_arr.shape:
    _emit({"correct": False, "shape_mismatch": [list(nki_arr.shape), list(ref_arr.shape)]})
    print("error: output shape mismatch", file=sys.stderr); sys.exit(9)

diff = np.abs(nki_arr.astype(np.float64) - ref_arr.astype(np.float64))
max_abs = float(diff.max())
denom = np.maximum(np.abs(ref_arr.astype(np.float64)), 1e-12)
rel = diff / denom
max_rel = float(rel.max())
mismatched = int(np.sum(diff > (atol + rtol * np.abs(ref_arr.astype(np.float64)))))
total = int(diff.size)
correct = bool(np.allclose(nki_arr, ref_arr, atol=atol, rtol=rtol))
_emit({
    "correct": correct,
    "max_abs_error": max_abs,
    "max_rel_error": max_rel,
    "mismatched_elements": mismatched,
    "total_elements": total,
})
sys.exit(0 if correct else 0)  # we report the result via the JSON line; non-zero is reserved for crashes
'''
