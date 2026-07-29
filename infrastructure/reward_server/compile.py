"""Compile a candidate NKI kernel by importing it and running it on the local
NeuronDevice via @nki.baremetal.

NKI kernels are JIT-compiled at call time — there is no separate `neuronx-cc
compile` step the way there is for HLO. To validate that the agent's emitted
source actually compiles, we:

  1. Write the source to a tempdir.
  2. Import it in a child process (so a segfault in the Neuron runtime
     can't take the reward server down).
  3. Generate dummy input tensors from `input_specs`.
  4. Wrap the kernel in @nki.baremetal and call it once.
  5. Capture stdout / stderr, returncode, and the .neff path the runtime
     wrote out.

Hard rules applied to caller-supplied source before we let it run:
  - source must be a string, <= 256 KB
  - kernel_name must be a Python identifier (used as filename + import target)
  - input_specs entries must have shape (list of ints) and dtype (str)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from reward_server.sandbox_env import build_subprocess_env

MAX_SOURCE_BYTES = 256 * 1024
_TMP_PATH_RE = re.compile(r"/tmp/[A-Za-z0-9_/.\-]+")  # nosec B108 — regex literal used to redact /tmp paths from log output, not an insecure tempfile call
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_DTYPE_RE = re.compile(r"^(float16|float32|bfloat16|int8|int16|int32)$")
_DEFAULT_INPUT = [{"shape": [128, 256], "dtype": "float32"}]


def _sanitize(line: str) -> str:
    return _TMP_PATH_RE.sub("<tmp>", line)


def compile_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("kernel_source")
    if not isinstance(source, str):
        return _err("kernel_source must be a string")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return _err(f"kernel_source exceeds {MAX_SOURCE_BYTES} bytes")

    kernel_name = payload.get("kernel_name", "nki_kernel")
    if not isinstance(kernel_name, str) or not _IDENT_RE.match(kernel_name):
        return _err("kernel_name must be a Python identifier")

    input_specs = payload.get("input_specs") or _DEFAULT_INPUT
    if not isinstance(input_specs, list) or not input_specs:
        return _err("input_specs must be a non-empty list")
    for spec in input_specs:
        if not isinstance(spec, dict):
            return _err("each input_spec must be an object")
        if not isinstance(spec.get("shape"), list) or not all(
            isinstance(d, int) and d > 0 for d in spec["shape"]
        ):
            return _err("input_spec.shape must be a list of positive ints")
        if not isinstance(spec.get("dtype"), str) or not _DTYPE_RE.match(spec["dtype"]):
            return _err("input_spec.dtype must be one of float16/float32/bfloat16/int*")

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="nki-compile-") as tmpdir:
        src_path = Path(tmpdir) / f"{kernel_name}.py"
        src_path.write_text(source, encoding="utf-8")
        neff_path = Path(tmpdir) / f"{kernel_name}.neff"
        runner_path = Path(tmpdir) / "runner.py"
        runner_path.write_text(_RUNNER_TEMPLATE, encoding="utf-8")

        env = build_subprocess_env(
            {
                "NKI_KERNEL_FILE": str(src_path),
                "NKI_KERNEL_NAME": kernel_name,
                "NKI_INPUT_SPECS": json.dumps(input_specs),
                "NKI_NEFF_PATH": str(neff_path),
            }
        )

        try:
            r = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return _err("compile timeout (>300s)", elapsed=time.time() - start)

        success = r.returncode == 0 and neff_path.exists()
        if success:
            # Request-scoped filename: the reward server runs with
            # multiple gunicorn workers (-w 4), so two concurrent requests
            # compiling the same kernel_name would otherwise both write to
            # Path(REWARD_NEFF_DIR)/f"{kernel_name}.neff" and race — one
            # request's persisted NEFF silently corrupting/overwriting the
            # other's. Suffixing with a per-request UUID makes collisions
            # impossible regardless of worker count, at the cost of the
            # persisted filename no longer being predictable from kernel_name
            # alone (callers must use the returned neff_path, which they
            # already do — nothing in this codebase re-derives this path).
            persisted = (
                Path(os.environ.get("REWARD_NEFF_DIR", "/tmp"))
                / f"{kernel_name}.{uuid.uuid4().hex}.neff"
            )
            persisted.write_bytes(neff_path.read_bytes())
        else:
            persisted = None

        stdout_lines = (r.stdout or "").splitlines()
        stderr_lines = (r.stderr or "").splitlines()
        merged = stdout_lines + stderr_lines
        errors = [_sanitize(line) for line in merged if "error" in line.lower()]
        warnings = [_sanitize(line) for line in merged if "warning" in line.lower()]
        return {
            "success": success,
            "neff_path": str(persisted) if persisted else None,
            "returncode": r.returncode,
            "errors": errors[-30:],
            "warnings": warnings[-30:],
            "stderr_tail": [_sanitize(line) for line in stderr_lines[-20:]],
            "compile_time_s": round(time.time() - start, 2),
        }


def _err(msg: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "success": False,
        "neff_path": None,
        "returncode": None,
        "errors": [msg],
        "warnings": [],
        "stderr_tail": [],
        "compile_time_s": round(elapsed, 2),
    }


# Child process: imports the source and JIT-compiles via @nki.baremetal.
# Runs in its own subprocess so a Neuron runtime crash can't take down the
# parent reward server.
_RUNNER_TEMPLATE = '''\
import importlib.util, json, os, sys, traceback
import numpy as np

src_path  = os.environ["NKI_KERNEL_FILE"]
kname     = os.environ["NKI_KERNEL_NAME"]
specs     = json.loads(os.environ["NKI_INPUT_SPECS"])
neff_path = os.environ["NKI_NEFF_PATH"]

try:
    import neuronxcc.nki as nki
except Exception as e:
    print(f"error: neuronxcc.nki import failed: {e}", file=sys.stderr)
    sys.exit(2)

try:
    spec = importlib.util.spec_from_file_location(kname, src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception:
    print("error: source import failed", file=sys.stderr)
    traceback.print_exc()
    sys.exit(3)

if not hasattr(mod, "nki_kernel"):
    print("error: source defines no nki_kernel function", file=sys.stderr)
    sys.exit(4)

DTYPE = {
    "float16": np.float16, "float32": np.float32, "bfloat16": np.float32,
    "int8": np.int8, "int16": np.int16, "int32": np.int32,
}
inputs = []
rng = np.random.default_rng(42)
for s in specs:
    arr = rng.standard_normal(size=tuple(s["shape"])).astype(DTYPE[s["dtype"]])
    inputs.append(arr)

try:
    bm = nki.baremetal(save_neff_name=neff_path)(mod.nki_kernel)
    out = bm(*inputs)
    print(f"compile_ok: output_shape={getattr(out, 'shape', None)}")
except Exception:
    print("error: baremetal compile/run failed", file=sys.stderr)
    traceback.print_exc()
    sys.exit(5)
'''
