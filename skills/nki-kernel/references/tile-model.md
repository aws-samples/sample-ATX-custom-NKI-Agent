# NKI tile model

NKI is an explicit tile-based programming model. The programmer declares tile shapes, loads tiles from HBM into on-chip SRAM, computes on tiles, and stores results back. This file is the reference for getting tile shapes right.

## Memory hierarchy

```
HBM   (host-visible, large, slow)
 │  nl.load
 ▼
SBUF  (on-chip scratch, ~24 MB on Trn1)
 │  Vector / Scalar engine ops
 ▼
PSUM  (matmul accumulator, ~2 MB on Trn1)
 │  nl.matmul / nl.add for accumulation
 ▼
SBUF  (back to scratch)
 │  nl.store
 ▼
HBM
```

Rules:
- All compute reads from SBUF or PSUM, never from HBM.
- Matmul writes to PSUM. Pointwise ops write to SBUF.
- You must explicitly stage data through SBUF; there is no implicit caching.

## Tile shape rules

Every tile has two dimensions: a **partition dimension** (P) and a **free dimension** (F).

- `TILE_P = 128` — exactly 128, always. This corresponds to the 128 partitions of the SBUF array. Anything else either won't compile or wastes hardware.
- `TILE_F` — flexible, but must fit in SBUF. Common choices: 128, 256, 512, 1024.

For a `(M, N)` tensor, the typical loop structure is:

```python
TILE_M = 128
TILE_N = 512

for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
        idx_m = m * TILE_M
        idx_n = n * TILE_N
        tile = nl.load(x[idx_m:idx_m+TILE_M, idx_n:idx_n+TILE_N])
        # compute on tile
        nl.store(out[idx_m:idx_m+TILE_M, idx_n:idx_n+TILE_N], tile)
```

## Index expressions

NKI's tile indexing is restrictive:

- Affine in loop variables — `m * TILE_M + r` where `r` is a constant or a small affine expression.
- No data-dependent indexing — you can't index a tile with the value of another tile.
- Sliced loads must be statically sized — `x[m:m+TILE_M, ...]` not `x[start:end, ...]` where `start, end` are runtime values.

## Common shape mistakes

| Mistake | Why it breaks |
|---|---|
| `TILE_M = 64` | Wastes half the SBUF partitions; some ops won't legalize. |
| 3D input `(B, M, N)` | NKI kernels expect 2D `(M, N)` tiles. Flatten the batch into M, or loop over batches at the host. |
| `nl.load` with a non-contiguous slice | Stride-1 along the free dimension is required. |
| Reusing a tile name across loop iterations without `nl.zeros_like` | The compiler may merge buffers in unsafe ways. |
| Output without `nl.shared_hbm` | The output tensor must live in shared HBM and be returned. |
