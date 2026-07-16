# NKI layout and signature rules

These rules are non-negotiable. Violating any of them produces a kernel that either fails to compile, fails to verify, or wastes the hardware.

## Function signature

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x, y):
    M, N = x.shape

    result = nl.ndarray(
        shape=(M, N),
        dtype=x.dtype,
        buffer=nl.shared_hbm,
    )

    # ... tile loop writing into result ...

    return result
```

Required:

1. **Function name is `nki_kernel`.** The agent's verify step looks up this exact symbol.
2. **`@nki.jit` decorator.** Without it, the function runs in eager Python.
3. **2D inputs.** `x.shape` is `(M, N)`. Never `(B, M, N)`.
4. **Output via `nl.ndarray(..., buffer=nl.shared_hbm)`.** Allocating with `nl.zeros` or returning a SBUF tile won't reach the host.
5. **Explicit `return result`.** The decorator does not infer outputs from side effects.

## Why TILE_M = 128

The Trainium NeuronCore SBUF is organized as 128 parallel partitions. A tile occupies one row per partition; any tile with `TILE_P != 128` either:

- leaves partitions idle (wasted compute), or
- forces the compiler to legalize via padding (which often fails).

For inputs where `M < 128`, pad to 128 at the host or process within a larger tile. For `M > 128`, loop over `M // 128` tiles.

## Operator availability

The following ops do **not** exist as `nl.<op>` and must be implemented manually:

| Want | Don't write | Write instead |
|---|---|---|
| sigmoid | `nl.sigmoid(x)` | `1 / (1 + nl.exp(-x))` via `nl.exp`, `nl.add`, `nl.divide` |
| relu | `nl.relu(x)` | `nl.maximum(x, 0)` |
| tanh | `nl.tanh(x)` | `(nl.exp(x) - nl.exp(-x)) / (nl.exp(x) + nl.exp(-x))` |
| gelu | `nl.gelu(x)` | exact gelu via erf approximation, or tanh approximation |
| sqrt | `nl.sqrt(x)` | `nl.power(x, 0.5)` or rsqrt+reciprocal |
| pow | `nl.pow(x, p)` | `nl.exp(p * nl.log(x))` for scalar p |

Always verify with `nki_skill_lookup` that an op exists before emitting it.

## dtype rules

- Inputs and outputs match dtype unless the task explicitly mixes.
- PSUM accumulation is always fp32 regardless of input dtype.
- bf16 inputs are common; cast to fp32 for accumulation, cast back to bf16 on store.

## What to do when the input is 3D

The customer hands you a `(B, M, N)` tensor. Your options:

1. **Flatten** — reshape to `(B*M, N)`, run the 2D kernel, reshape back. Cheapest, works when there's no per-batch state.
2. **Loop at host** — have the calling code iterate over the batch dimension and call the kernel per-batch. Right answer when there is per-batch reduction (e.g., layernorm).

Pick at the planning step, not at code-gen time. Document the choice in `.nki-agent/runs/<kernel-id>/notes.md`.
