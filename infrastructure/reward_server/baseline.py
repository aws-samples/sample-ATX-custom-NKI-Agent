"""Run the PyTorch reference under `torch.compile(backend="openxla")` on Trn1.

This is the baseline column for the blog: the PyTorch-2 native path on
Neuron SDK 2.x. `torch_neuronx` registers a PJRT runtime under `torch_xla`,
so the customer-facing API is `torch.compile(model, backend="openxla")`
running on `xm.xla_device()`. This is the modern path — not the legacy
`torch_neuronx.trace` JIT.

Same percentile schema as profile.py so blog tables can be apples-to-apples.

Caller payload:
  reference_source str — must define `reference(*args)` (or expose `Model`
                         with `forward(self, *args)`); we eval on Trn1.
  input_specs      list — [{"shape": [...], "dtype": "float32"|...}]
  warmup_runs      int  — default 5
  measure_runs     int  — default 50
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
_DTYPE_RE = re.compile(r"^(float16|float32|bfloat16|int8|int16|int32)$")


def _sanitize(line: str) -> str:
    return _TMP_PATH_RE.sub("<tmp>", line)


def _sanitize_block(s: str) -> str:
    return "\n".join(_sanitize(line) for line in s.splitlines())


def baseline_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("reference_source")
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE_BYTES:
        return _err("reference_source missing or too large")

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

    try:
        warmup_runs = int(payload.get("warmup_runs", 5))
        measure_runs = int(payload.get("measure_runs", 50))
    except (TypeError, ValueError):
        return _err("warmup_runs / measure_runs must be ints")
    if not (0 <= warmup_runs <= 100) or not (1 <= measure_runs <= 1000):
        return _err("warmup_runs in [0,100], measure_runs in [1,1000]")

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="nki-baseline-") as tmpdir:
        rpath = Path(tmpdir) / "reference.py"
        runner = Path(tmpdir) / "runner.py"
        rpath.write_text(source, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")

        env = build_subprocess_env(
            {
                "NKI_REFERENCE_FILE": str(rpath),
                "NKI_INPUT_SPECS": json.dumps(input_specs),
                "NKI_WARMUP": str(warmup_runs),
                "NKI_MEASURE": str(measure_runs),
            }
        )

        try:
            r = subprocess.run(
                [sys.executable, str(runner)],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return _err("baseline timeout (>900s)", elapsed=time.time() - start)

        result_line = ""
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("__BASELINE__"):
                result_line = ln[len("__BASELINE__"):]
        try:
            metrics = json.loads(result_line) if result_line else {}
        except json.JSONDecodeError:
            metrics = {}

        if r.returncode != 0 or not metrics:
            return {
                "success": False,
                "backend": "torch.compile(backend=openxla)+torch_neuronx",
                "latency_mean_us": None,
                "latency_median_us": None,
                "latency_p99_us": None,
                "latency_min_us": None,
                "latency_max_us": None,
                "throughput_ops_per_s": None,
                "warmup_runs": warmup_runs,
                "measure_runs": measure_runs,
                "baseline_time_s": round(time.time() - start, 2),
                "error": _sanitize_block((r.stderr or "") + "\n" + (r.stdout or ""))[-1500:],
            }

        return {
            "success": True,
            "backend": "torch.compile(backend=openxla)+torch_neuronx",
            "latency_mean_us": metrics.get("mean_us"),
            "latency_median_us": metrics.get("p50_us"),
            "latency_p99_us": metrics.get("p99_us"),
            "latency_min_us": metrics.get("min_us"),
            "latency_max_us": metrics.get("max_us"),
            "throughput_ops_per_s": metrics.get("throughput_ops_s"),
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "baseline_time_s": round(time.time() - start, 2),
            "error": None,
        }


def _err(msg: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "success": False,
        "backend": "torch.compile(backend=openxla)+torch_neuronx",
        "latency_mean_us": None,
        "latency_median_us": None,
        "latency_p99_us": None,
        "latency_min_us": None,
        "latency_max_us": None,
        "throughput_ops_per_s": None,
        "warmup_runs": None,
        "measure_runs": None,
        "baseline_time_s": round(elapsed, 2),
        "error": msg,
    }


# Runner: lives in subprocess so a torch-neuronx import explosion or a Neuron
# runtime fault can't take the parent reward server down. Loads the supplied
# reference module, wraps it in torch.compile(backend="openxla") on the
# torch_xla / PJRT runtime that torch_neuronx provides — this is the
# PyTorch-2 native path on Neuron SDK 2.x (NOT torch_neuronx.trace).
# Runs warmup + timed iterations on the local NeuronCore, emits percentiles
# to stdout under a __BASELINE__ prefix.
_RUNNER = '''\
import importlib.util, json, os, sys, time, traceback
import numpy as np

rpath = os.environ["NKI_REFERENCE_FILE"]
specs = json.loads(os.environ["NKI_INPUT_SPECS"])
warmup = int(os.environ["NKI_WARMUP"])
measure = int(os.environ["NKI_MEASURE"])

def _emit(m):
    sys.stdout.write("__BASELINE__" + json.dumps(m) + "\\n")
    sys.stdout.flush()

try:
    import torch
except Exception as e:
    print(f"error: torch import failed: {e}", file=sys.stderr); sys.exit(2)

# torch-neuronx 2.x exposes a `torch.compile(backend="neuronx")` integration.
# This is the PyTorch-2 native path; NOT torch_xla / PJRT.
try:
    import torch_neuronx  # noqa: F401  — registers the backend.
except Exception as e:
    print(f"error: torch_neuronx import failed: {e}", file=sys.stderr); sys.exit(2)

DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16,
         "int8": torch.int8, "int16": torch.int16, "int32": torch.int32}
torch.manual_seed(42)
inputs = [torch.randn(tuple(s["shape"]), dtype=DTYPE[s["dtype"]]) if s["dtype"].startswith("float") or s["dtype"] == "bfloat16"
          else torch.randint(-128, 128, tuple(s["shape"]), dtype=DTYPE[s["dtype"]])
          for s in specs]

try:
    spec = importlib.util.spec_from_file_location("reference", rpath)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
except Exception:
    print("error: reference import failed", file=sys.stderr); traceback.print_exc(); sys.exit(3)

# Pick the callable: prefer reference(*args), fall back to Model().forward(*args).
fn = None
if hasattr(mod, "reference") and callable(mod.reference):
    fn = mod.reference
elif hasattr(mod, "Model"):
    Model = mod.Model
    instance = Model() if not isinstance(Model, torch.nn.Module) else Model
    fn = lambda *a: instance(*a)
elif hasattr(mod, "forward") and callable(mod.forward):
    fn = mod.forward
else:
    print("error: reference defines no `reference`, `Model`, or `forward`", file=sys.stderr); sys.exit(4)

# Move inputs to xla device first (torch_neuronx provides PJRT under torch_xla;
# the device is the NeuronCore). Then torch.compile(backend="openxla") fuses
# the graph through the Neuron compiler. This is the PyTorch-2 native baseline
# customers run today — NOT the legacy torch_neuronx.trace JIT path.
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    on_dev = [t.to(device) for t in inputs]
except Exception:
    print("error: torch_xla device init failed", file=sys.stderr); traceback.print_exc(); sys.exit(5)

try:
    compiled = torch.compile(fn, backend="openxla")
except Exception:
    print("error: torch.compile(backend=openxla) failed", file=sys.stderr); traceback.print_exc(); sys.exit(5)

# Warmup: also primes any tracing / NEFF compile inside torch.compile.
try:
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            _ = compiled(*on_dev)
            xm.mark_step(); xm.wait_device_ops()
except Exception:
    print("error: warmup failed", file=sys.stderr); traceback.print_exc(); sys.exit(6)

# Timed iterations. mark_step + wait_device_ops are inside the timing window
# so the measurement reflects end-to-end device latency, not lazy-graph
# enqueue. This is what a customer would observe from their training loop.
samples_us = []
try:
    with torch.no_grad():
        for _ in range(measure):
            t0 = time.perf_counter()
            _ = compiled(*on_dev)
            xm.mark_step(); xm.wait_device_ops()
            samples_us.append((time.perf_counter() - t0) * 1e6)
except Exception:
    print("error: timed loop failed", file=sys.stderr); traceback.print_exc(); sys.exit(7)

samples_us.sort()
n = len(samples_us)
def _pct(p):
    if n == 0: return 0.0
    k = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
    return float(samples_us[k])

p50 = _pct(50)
mean = sum(samples_us) / n if n else 0.0
ops_s = (1e6 / mean) if mean > 0 else None

_emit({
    "min_us": _pct(0),
    "p50_us": p50,
    "mean_us": mean,
    "p90_us": _pct(90),
    "p99_us": _pct(99),
    "max_us": _pct(100),
    "throughput_ops_s": ops_s,
})
'''
