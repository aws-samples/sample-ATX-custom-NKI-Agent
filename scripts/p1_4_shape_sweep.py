"""P1.4 — shape sweep across (128, 256), (128, 4096), (1024, 4096).

3 hand-picked targets (softmax / rms_norm / layer_norm), 3 shapes each,
2 backends each (NKI kernel via op=profile + PyTorch reference via
op=baseline). 18 reward-server POSTs total, --max-workers 1.

Output:
  results/p1_4_shape_sweep_<date>.json
    [
      {target, shape, dtype, kernel:{p50_us, p99_us, ...},
                                baseline:{p50_us, p99_us, ...}, speedup_p50},
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# ---------------------------------------------------------------------------
# Targets — each: NKI kernel source + PyTorch reference source. Both must work
# on (M, N) 2D inputs (M%128==0). The reference must define `reference(x)`.
# ---------------------------------------------------------------------------

SOFTMAX_KERNEL = '''\
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        i_m = nl.arange(TILE_M)[:, None]
        i_n = nl.arange(N)[None, :]
        x_tile = nl.load(x[m * TILE_M + i_m, i_n])
        # row max for numerical stability
        row_max = nl.max(x_tile, axis=[1], keepdims=True)
        shifted = nl.subtract(x_tile, row_max)
        exp_x = nl.exp(shifted)
        row_sum = nl.sum(exp_x, axis=[1], keepdims=True)
        out = nl.divide(exp_x, row_sum)
        nl.store(result[m * TILE_M + i_m, i_n], value=out)
    return result
'''

SOFTMAX_REF = '''\
import torch
import torch.nn.functional as F

def reference(x):
    return F.softmax(x, dim=-1)
'''

RMSNORM_KERNEL = '''\
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    eps = 1e-6
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        i_m = nl.arange(TILE_M)[:, None]
        i_n = nl.arange(N)[None, :]
        x_tile = nl.load(x[m * TILE_M + i_m, i_n])
        sq = nl.multiply(x_tile, x_tile)
        mean_sq = nl.sum(sq, axis=[1], keepdims=True)
        inv_n = nl.full((TILE_M, 1), 1.0 / float(N), dtype=x_tile.dtype, buffer=nl.sbuf)
        var = nl.multiply(mean_sq, inv_n)
        eps_tile = nl.full((TILE_M, 1), eps, dtype=x_tile.dtype, buffer=nl.sbuf)
        var_eps = nl.add(var, eps_tile)
        inv = nl.rsqrt(var_eps)
        out = nl.multiply(x_tile, inv)
        nl.store(result[m * TILE_M + i_m, i_n], value=out)
    return result
'''

# Reference matches the kernel: weight=1, no scale (keeps it apples-to-apples
# with the kernel's compute; introducing a learned weight would just be a
# pointwise multiply that both sides do equally).
RMSNORM_REF = '''\
import torch

def reference(x):
    eps = 1e-6
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)
'''

LAYERNORM_KERNEL = '''\
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    eps = 1e-5
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        i_m = nl.arange(TILE_M)[:, None]
        i_n = nl.arange(N)[None, :]
        x_tile = nl.load(x[m * TILE_M + i_m, i_n])
        # mean
        row_sum = nl.sum(x_tile, axis=[1], keepdims=True)
        inv_n = nl.full((TILE_M, 1), 1.0 / float(N), dtype=x_tile.dtype, buffer=nl.sbuf)
        mean = nl.multiply(row_sum, inv_n)
        centered = nl.subtract(x_tile, mean)
        # variance
        sq = nl.multiply(centered, centered)
        sq_sum = nl.sum(sq, axis=[1], keepdims=True)
        var = nl.multiply(sq_sum, inv_n)
        eps_tile = nl.full((TILE_M, 1), eps, dtype=x_tile.dtype, buffer=nl.sbuf)
        inv = nl.rsqrt(nl.add(var, eps_tile))
        out = nl.multiply(centered, inv)
        nl.store(result[m * TILE_M + i_m, i_n], value=out)
    return result
'''

LAYERNORM_REF = '''\
import torch
import torch.nn.functional as F

def reference(x):
    return F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=1e-5)
'''

TARGETS = [
    {"name": "softmax",   "kernel": SOFTMAX_KERNEL,   "reference": SOFTMAX_REF},
    {"name": "rms_norm",  "kernel": RMSNORM_KERNEL,   "reference": RMSNORM_REF},
    {"name": "layer_norm","kernel": LAYERNORM_KERNEL, "reference": LAYERNORM_REF},
]

SHAPES = [(128, 256), (128, 4096), (1024, 4096)]
DTYPE = "float32"


# ---------------------------------------------------------------------------
# Reward-server client
# ---------------------------------------------------------------------------

def _sigv4_headers(method: str, url: str, body: bytes, region: str) -> dict:
    """Sign a request with the caller's own AWS credential chain against
    the `execute-api` service.

    This signing is currently a no-op against the reward server: the private
    API Gateway that used to verify these signatures was removed (see
    reward_server_cdk/lib/reward-server-stack.ts and reward_server/auth.py) —
    the reward server's actual (and only) access control is network position,
    reachable solely from the AgentCore runtime's security group inside the
    VPC. Calling this against the reward server requires a real network path
    into that VPC; the AWS credentials below are still needed to construct a
    valid request but do not currently gate anything on the receiving end.
    This is an open item pending a decision on what MCP-direct access to the
    reward server should actually require."""
    creds = boto3.Session(region_name=region).get_credentials()
    if creds is None:
        raise RuntimeError(
            "no AWS credentials found; configure env vars, ~/.aws/credentials, "
            "or IAM Identity Center. Note: this only produces a well-formed "
            "SigV4-signed request; it does not by itself grant reward-server "
            "access, which is gated by network position, not IAM (see "
            "reward_server/auth.py)."
        )
    req = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(creds, "execute-api", region).add_auth(req)
    return dict(req.headers)


def post(url: str, payload: dict, region: str, timeout: int = 1200) -> dict:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) reward-server URL: {url}")
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    headers.update(_sigv4_headers("POST", url, body, region))
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 — url scheme validated above
        return json.loads(r.read().decode())


def profile_kernel(url: str, region: str, source: str, shape, dtype: str,
                   warmup: int = 5, measure: int = 50) -> dict:
    return post(url, {
        "op": "profile",
        "kernel_source": source,
        "input_specs": [{"shape": list(shape), "dtype": dtype}],
        "warmup_runs": warmup,
        "measure_runs": measure,
    }, region)


def baseline_run(url: str, region: str, source: str, shape, dtype: str,
                 warmup: int = 5, measure: int = 50) -> dict:
    return post(url, {
        "op": "baseline",
        "reference_source": source,
        "input_specs": [{"shape": list(shape), "dtype": dtype}],
        "warmup_runs": warmup,
        "measure_runs": measure,
    }, region)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-url", default=os.environ.get(
        "REWARD_URL", "http://REWARD_SERVER_IP:5050/reward"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=50)
    args = parser.parse_args()

    out_path = args.output or (
        "results/p1_4_shape_sweep_" + datetime.now().strftime("%Y-%m-%d_%H%M") + ".json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()

    for tgt in TARGETS:
        for shape in SHAPES:
            row = {
                "target": tgt["name"],
                "shape": list(shape),
                "dtype": DTYPE,
                "kernel": None,
                "baseline": None,
                "speedup_p50": None,
                "elapsed_s": None,
            }
            t0 = time.time()
            print(f"\n[{tgt['name']:10s} {tuple(shape)}] profile kernel…",
                  flush=True)
            try:
                row["kernel"] = profile_kernel(
                    args.reward_url, args.region,
                    tgt["kernel"], shape, DTYPE,
                    warmup=args.warmup, measure=args.measure,
                )
                k_ok = row["kernel"].get("success")
                k_p50 = row["kernel"].get("latency_median_us")
                print(f"  kernel:   success={k_ok}  p50={k_p50}us")
            except Exception as e:
                row["kernel"] = {"success": False, "error": f"client: {e}"}
                print(f"  kernel:   client error: {e}")

            print(f"[{tgt['name']:10s} {tuple(shape)}] run baseline…",
                  flush=True)
            try:
                row["baseline"] = baseline_run(
                    args.reward_url, args.region,
                    tgt["reference"], shape, DTYPE,
                    warmup=args.warmup, measure=args.measure,
                )
                b_ok = row["baseline"].get("success")
                b_p50 = row["baseline"].get("latency_median_us")
                print(f"  baseline: success={b_ok}  p50={b_p50}us")
            except Exception as e:
                row["baseline"] = {"success": False, "error": f"client: {e}"}
                print(f"  baseline: client error: {e}")

            k = (row["kernel"] or {}).get("latency_median_us")
            b = (row["baseline"] or {}).get("latency_median_us")
            if isinstance(k, (int, float)) and isinstance(b, (int, float)) and k > 0:
                row["speedup_p50"] = round(b / k, 3)

            row["elapsed_s"] = round(time.time() - t0, 1)
            rows.append(row)

            # Incremental save so a mid-run failure doesn't lose data.
            with open(out_path, "w") as f:
                json.dump({
                    "started_at": datetime.fromtimestamp(t_start).isoformat(),
                    "reward_url": args.reward_url,
                    "warmup_runs": args.warmup,
                    "measure_runs": args.measure,
                    "rows": rows,
                }, f, indent=2)

    print("\n=== summary ===")
    print(f"{'target':12s} {'shape':16s} {'kernel p50':>12s} {'base p50':>12s} {'speedup':>10s}")
    for r in rows:
        kp = (r.get("kernel") or {}).get("latency_median_us")
        bp = (r.get("baseline") or {}).get("latency_median_us")
        sp = r.get("speedup_p50")
        kp_s = f"{kp:.1f}" if isinstance(kp, (int, float)) else "—"
        bp_s = f"{bp:.1f}" if isinstance(bp, (int, float)) else "—"
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "—"
        print(f"{r['target']:12s} {str(tuple(r['shape'])):16s} "
              f"{kp_s:>12s} {bp_s:>12s} {sp_s:>10s}")

    print(f"\nwrote {out_path}  ({round(time.time() - t_start, 1)}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
