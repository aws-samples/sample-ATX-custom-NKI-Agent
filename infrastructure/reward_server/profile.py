"""Profile a NKI kernel via @nki.benchmark on the local NeuronDevice.

Uses the native `@nki.benchmark(warmup=N, iters=M)` decorator, which invokes
neuron-bench under the hood — one NEFF load, M timed runs, then teardown.
This avoids the NRT ret=13 we saw when looping `nki.baremetal` calls and
re-loading the NEFF every iteration.

Caller payload:
  kernel_source str — must define `nki_kernel`
  input_specs   list — [{"shape": [...], "dtype": "float32"|...}]
  warmup_runs   int  — default 5
  measure_runs  int  — default 50
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


def profile_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("kernel_source")
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE_BYTES:
        return _err("kernel_source missing or too large")
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

    try:
        warmup_runs = int(payload.get("warmup_runs", 5))
        measure_runs = int(payload.get("measure_runs", 50))
    except (TypeError, ValueError):
        return _err("warmup_runs / measure_runs must be ints")
    if not (0 <= warmup_runs <= 100) or not (1 <= measure_runs <= 1000):
        return _err("warmup_runs in [0,100], measure_runs in [1,1000]")

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="nki-profile-") as tmpdir:
        kpath = Path(tmpdir) / f"{kernel_name}.py"
        runner = Path(tmpdir) / "runner.py"
        kpath.write_text(source, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")

        env = build_subprocess_env(
            {
                "NKI_KERNEL_FILE": str(kpath),
                "NKI_KERNEL_NAME": kernel_name,
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
            return _err("profile timeout (>900s)", elapsed=time.time() - start)

        result_line = ""
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("__PROFILE__"):
                result_line = ln[len("__PROFILE__"):]
        try:
            metrics = json.loads(result_line) if result_line else {}
        except json.JSONDecodeError:
            metrics = {}

        if r.returncode != 0 or not metrics:
            return {
                "success": False,
                "latency_mean_us": None,
                "latency_median_us": None,
                "latency_p99_us": None,
                "latency_min_us": None,
                "latency_max_us": None,
                "throughput_ops_per_s": None,
                "flops_utilization": None,
                "warmup_runs": warmup_runs,
                "measure_runs": measure_runs,
                "profile_time_s": round(time.time() - start, 2),
                "error": _sanitize_block((r.stderr or "") + "\n" + (r.stdout or ""))[-1500:],
            }

        return {
            "success": True,
            "latency_mean_us": metrics.get("mean_us"),
            "latency_median_us": metrics.get("p50_us"),
            "latency_p99_us": metrics.get("p99_us"),
            "latency_min_us": metrics.get("min_us"),
            "latency_max_us": metrics.get("max_us"),
            "throughput_ops_per_s": metrics.get("throughput_ops_s"),
            "flops_utilization": None,  # neuron-profile parsing is a follow-up
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "profile_time_s": round(time.time() - start, 2),
            "error": None,
        }


def _err(msg: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "success": False,
        "latency_mean_us": None,
        "latency_median_us": None,
        "latency_p99_us": None,
        "latency_min_us": None,
        "latency_max_us": None,
        "throughput_ops_per_s": None,
        "flops_utilization": None,
        "warmup_runs": None,
        "measure_runs": None,
        "profile_time_s": round(elapsed, 2),
        "error": msg,
    }


def _sanitize_block(s: str) -> str:
    return "\n".join(_sanitize(line) for line in s.splitlines())


_RUNNER = '''\
import importlib.util, json, os, sys, traceback
import numpy as np

kpath = os.environ["NKI_KERNEL_FILE"]
kname = os.environ["NKI_KERNEL_NAME"]
specs = json.loads(os.environ["NKI_INPUT_SPECS"])
warmup = int(os.environ["NKI_WARMUP"])
measure = int(os.environ["NKI_MEASURE"])

def _emit(m):
    sys.stdout.write("__PROFILE__" + json.dumps(m) + "\\n")
    sys.stdout.flush()

try:
    import neuronxcc.nki as nki
except Exception as e:
    print(f"error: nki import failed: {e}", file=sys.stderr); sys.exit(2)

DTYPE = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32,
         "int8": np.int8, "int16": np.int16, "int32": np.int32}
rng = np.random.default_rng(42)
inputs = [rng.standard_normal(size=tuple(s["shape"])).astype(DTYPE[s["dtype"]]) for s in specs]

try:
    spec = importlib.util.spec_from_file_location(kname, kpath)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
except Exception:
    print("error: kernel import failed", file=sys.stderr); traceback.print_exc(); sys.exit(3)

if not hasattr(mod, "nki_kernel"):
    print("error: source defines no nki_kernel function", file=sys.stderr); sys.exit(4)

# Native @nki.benchmark — neuron-bench loads the NEFF once and runs `iters`
# timed iterations on device, then returns a Benchmark object whose
# `nc_latency` exposes percentile getters.
try:
    bm = nki.benchmark(warmup=max(1, warmup), iters=measure)(mod.nki_kernel)
    bm(*inputs)
    lat = bm.benchmark_result.nc_latency
except Exception:
    print("error: benchmark loop failed", file=sys.stderr); traceback.print_exc(); sys.exit(5)

# nc_latency.get_latency_percentile takes int in [0,100]; values are in microseconds.
try:
    p0  = float(lat.get_latency_percentile(0))
    p50 = float(lat.get_latency_percentile(50))
    p90 = float(lat.get_latency_percentile(90))
    p99 = float(lat.get_latency_percentile(99))
    p100 = float(lat.get_latency_percentile(100))
except Exception:
    print("error: percentile extraction failed", file=sys.stderr); traceback.print_exc(); sys.exit(6)

# Mean: API doesn't expose mean directly, approximate from p50.
mean_us = p50
ops_s = (1e6 / mean_us) if mean_us > 0 else None

_emit({
    "min_us": p0,
    "p50_us": p50,
    "mean_us": mean_us,
    "p90_us": p90,
    "p99_us": p99,
    "max_us": p100,
    "throughput_ops_s": ops_s,
})
'''
