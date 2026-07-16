"""Input-validation tests for the reward-server handlers.

These cover the security-critical path: the reward server runs model-generated
code, so every handler must reject malformed / oversized / mistyped payloads
*before* spawning a subprocess. All tests here exercise the validation branches
only — they never reach the Neuron toolchain, so they run anywhere.
"""

from __future__ import annotations

import pytest

from reward_server.baseline import baseline_kernel
from reward_server.compile import compile_kernel
from reward_server.verify import verify_kernel

VALID_SPEC = [{"shape": [128, 256], "dtype": "float32"}]
GOOD_SRC = "import neuronxcc.nki as nki\n@nki.jit\ndef nki_kernel(x):\n    return x\n"


# --------------------------------------------------------------------------- #
# compile_kernel
# --------------------------------------------------------------------------- #
def test_compile_rejects_non_string_source() -> None:
    r = compile_kernel({"kernel_source": 123})
    assert r["success"] is False
    assert "string" in r["errors"][0]


def test_compile_rejects_oversized_source() -> None:
    r = compile_kernel({"kernel_source": "x" * (256 * 1024 + 1)})
    assert r["success"] is False
    assert "exceeds" in r["errors"][0]


@pytest.mark.parametrize("name", ["has space", "1leading", "bad-hyphen", "a" * 65, ""])
def test_compile_rejects_bad_kernel_name(name: str) -> None:
    r = compile_kernel({"kernel_source": GOOD_SRC, "kernel_name": name})
    assert r["success"] is False
    assert "identifier" in r["errors"][0]


def test_compile_persists_neff_with_unique_per_request_filename(
    monkeypatch, tmp_path
) -> None:
    """The reward server runs gunicorn with multiple workers, so two
    concurrent requests compiling the same kernel_name must not collide on
    the persisted NEFF filename. Simulates a successful compile (no real
    Neuron toolchain involved) and asserts the persisted filename is
    request-scoped rather than derived from kernel_name alone."""
    import subprocess
    from pathlib import Path
    from unittest.mock import MagicMock

    neff_dir = tmp_path / "neff_out"
    neff_dir.mkdir()
    monkeypatch.setenv("REWARD_NEFF_DIR", str(neff_dir))

    def fake_run(args, **kwargs):
        # The runner would have written NKI_NEFF_PATH on success; simulate
        # that here since we aren't invoking the real toolchain.
        env = kwargs["env"]
        Path(env["NKI_NEFF_PATH"]).write_bytes(b"fake-neff-bytes")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "compile_ok: output_shape=(128, 256)"
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)

    r1 = compile_kernel({"kernel_source": GOOD_SRC, "kernel_name": "same_name"})
    r2 = compile_kernel({"kernel_source": GOOD_SRC, "kernel_name": "same_name"})

    assert r1["success"] is True
    assert r2["success"] is True
    assert r1["neff_path"] != r2["neff_path"]
    # Both persisted files must actually exist side-by-side, not have
    # overwritten each other.
    assert Path(r1["neff_path"]).exists()
    assert Path(r2["neff_path"]).exists()
    assert list(neff_dir.glob("same_name.*.neff")) and len(
        list(neff_dir.glob("same_name.*.neff"))
    ) == 2


@pytest.mark.parametrize(
    "specs",
    [
        "notalist",
        [],
        [{"shape": [128, 256]}],                 # missing dtype
        [{"dtype": "float32"}],                  # missing shape
        [{"shape": [0, 256], "dtype": "float32"}],   # non-positive dim
        [{"shape": [128, -1], "dtype": "float32"}],
        [{"shape": [128, 2.0], "dtype": "float32"}], # non-int dim
        [{"shape": [128, 256], "dtype": "float64"}], # dtype not allowlisted
        [{"shape": [128, 256], "dtype": "'; rm -rf /"}],
    ],
)
def test_compile_rejects_bad_input_specs(specs: object) -> None:
    r = compile_kernel({"kernel_source": GOOD_SRC, "input_specs": specs})
    assert r["success"] is False


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16", "int8", "int16", "int32"])
def test_compile_accepts_allowlisted_dtypes_through_validation(dtype: str) -> None:
    # Passes validation; the subprocess run will fail (no Neuron here), but the
    # returned error must NOT be one of the validation messages.
    r = compile_kernel(
        {"kernel_source": GOOD_SRC, "input_specs": [{"shape": [128, 256], "dtype": dtype}]}
    )
    joined = " ".join(r["errors"])
    assert "dtype must be" not in joined
    assert "shape must be" not in joined


# --------------------------------------------------------------------------- #
# verify_kernel
# --------------------------------------------------------------------------- #
def test_verify_rejects_missing_reference() -> None:
    r = verify_kernel({"kernel_source": GOOD_SRC, "input_specs": VALID_SPEC})
    assert r["correct"] is False
    assert "reference_source" in r["error"]


def test_verify_rejects_oversized_reference() -> None:
    r = verify_kernel(
        {
            "kernel_source": GOOD_SRC,
            "reference_source": "x" * (256 * 1024 + 1),
            "input_specs": VALID_SPEC,
        }
    )
    assert r["correct"] is False
    assert "too large" in r["error"]


def test_verify_rejects_bad_kernel_name() -> None:
    r = verify_kernel(
        {
            "kernel_source": GOOD_SRC,
            "reference_source": "def reference(x):\n    return x\n",
            "input_specs": VALID_SPEC,
            "kernel_name": "bad name",
        }
    )
    assert r["correct"] is False
    assert "identifier" in r["error"]


def test_verify_rejects_empty_specs() -> None:
    r = verify_kernel(
        {"kernel_source": GOOD_SRC, "reference_source": "x", "input_specs": []}
    )
    assert r["correct"] is False
    assert "input_specs" in r["error"]


# --------------------------------------------------------------------------- #
# baseline_kernel
# --------------------------------------------------------------------------- #
def test_baseline_rejects_missing_reference() -> None:
    r = baseline_kernel({"input_specs": VALID_SPEC})
    assert r["success"] is False
    assert "reference_source" in r["error"]


@pytest.mark.parametrize(
    "warmup,measure",
    [(-1, 50), (101, 50), (5, 0), (5, 1001)],
)
def test_baseline_rejects_out_of_range_run_counts(warmup: int, measure: int) -> None:
    r = baseline_kernel(
        {
            "reference_source": "def reference(x):\n    return x\n",
            "input_specs": VALID_SPEC,
            "warmup_runs": warmup,
            "measure_runs": measure,
        }
    )
    assert r["success"] is False
    assert "warmup_runs" in r["error"] or "measure_runs" in r["error"]


def test_baseline_rejects_non_int_run_counts() -> None:
    r = baseline_kernel(
        {
            "reference_source": "def reference(x):\n    return x\n",
            "input_specs": VALID_SPEC,
            "warmup_runs": "five",
        }
    )
    assert r["success"] is False
