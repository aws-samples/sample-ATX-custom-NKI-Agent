# neuronx-cc compiler error decoder

When `nki_compile` fails, the model needs to know what the error means and what to change. Here's the decoder ring.

## "Partition dimension must be 128"

You used a tile shape with a partition dimension other than 128.

**Fix:** set `TILE_M = 128`. If the input has fewer than 128 rows, pad at the host or process within a larger fused tile.

## "Unsupported NKI builtin: nl.<op>"

The op you called doesn't exist in NKI.

**Fix:** check `references/layout-rules.md` for the manual implementation. Common offenders: `nl.sigmoid`, `nl.relu`, `nl.tanh`, `nl.gelu`, `nl.sqrt`, `nl.pow`.

## "Tile shape exceeds SBUF capacity"

Your tile is too large to fit in on-chip scratch.

**Fix:** shrink `TILE_F` (the free dimension). Halve it and re-compile. SBUF is ~24 MB on Trn1.

## "Non-affine index expression"

You tried to index a tile with a value that isn't a static affine expression of loop variables.

**Fix:** restructure so the index is `loop_var * TILE + constant`. Data-dependent indexing isn't supported.

## "Cannot infer dtype for reduction"

Reductions (`nl.sum`, `nl.max`, `nl.mean`) without an explicit dtype on bf16 inputs.

**Fix:** cast the input to fp32 for the reduction, then cast back. PSUM accumulation is always fp32 anyway.

## "Layout mismatch: expected 2D, got 3D"

You passed a 3D input.

**Fix:** flatten `(B, M, N)` to `(B*M, N)` at the host, or hoist the batch loop above the kernel call. See `references/layout-rules.md` § "What to do when the input is 3D".

## "Loop bounds must be statically known"

`nl.affine_range` with a runtime value.

**Fix:** make the range a compile-time constant or pass shape via the function signature so the compiler sees it.

## "Multiple writes to PSUM tile"

You wrote to the same PSUM accumulator from non-mergeable ops.

**Fix:** allocate a fresh `nl.zeros(..., buffer=nl.psum)` per matmul reduction loop. Don't share PSUM across unrelated matmuls.

## "neuronx-cc internal error" / no useful message

The compiler crashed with no actionable signal. This usually means an unsupported pattern that hit a code path with no diagnostic.

**Fix:** simplify aggressively. Remove fusion, split the kernel into smaller pieces, and re-compile. Then add complexity back. Surface to the user if this persists across 3+ attempts — it's a real compiler bug, not a model bug.

## What to feed back to the model on retry

Include in the next prompt:

1. The full compiler stderr (truncate to last 50 lines if very long).
2. A 1-line diagnosis from this file (e.g., "Error: partition dim != 128").
3. The candidate source that failed.
4. An instruction: *"fix the specific issue, do not rewrite the rest of the kernel."*

The agent's job on retry is local repair, not regeneration.
