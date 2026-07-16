# Common NKI op recipes

Reference implementations for ops the agent encounters most often. These are starting points, not final code — the agent should adapt shapes, dtypes, and tile sizes to the task.

## Pointwise: elementwise add

```python
@nki.jit
def nki_kernel(a, b):
    M, N = a.shape
    TILE_M = 128
    TILE_N = 512

    result = nl.ndarray(shape=(M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            ta = nl.load(a[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])
            tb = nl.load(b[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])
            tr = nl.add(ta, tb)
            nl.store(result[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], tr)

    return result
```

## Matmul (TensorEngine, PSUM)

```python
@nki.jit
def nki_kernel(a, b):
    # a: (M, K), b: (K, N) — output (M, N)
    M, K = a.shape
    _, N = b.shape
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512

    result = nl.ndarray(shape=(M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                ta = nl.load(a[m*TILE_M:(m+1)*TILE_M, k*TILE_K:(k+1)*TILE_K])
                tb = nl.load(b[k*TILE_K:(k+1)*TILE_K, n*TILE_N:(n+1)*TILE_N])
                acc += nl.matmul(ta, tb)
            tr = nl.cast(acc, dtype=a.dtype)
            nl.store(result[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], tr)

    return result
```

## Softmax (numerically stable, row-wise)

```python
@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    # softmax requires the whole row in SBUF; pick TILE_N = N if it fits

    result = nl.ndarray(shape=(M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        row = nl.load(x[m*TILE_M:(m+1)*TILE_M, :])
        row_max = nl.max(row, axis=1, keepdims=True)
        shifted = nl.subtract(row, row_max)
        exp = nl.exp(shifted)
        denom = nl.sum(exp, axis=1, keepdims=True)
        out = nl.divide(exp, denom)
        nl.store(result[m*TILE_M:(m+1)*TILE_M, :], out)

    return result
```

## RMSNorm (row-wise)

```python
@nki.jit
def nki_kernel(x, weight):
    M, N = x.shape
    TILE_M = 128
    EPS = 1e-6

    result = nl.ndarray(shape=(M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    w = nl.load(weight[:])  # (N,)

    for m in nl.affine_range(M // TILE_M):
        row = nl.load(x[m*TILE_M:(m+1)*TILE_M, :])
        sq = nl.multiply(row, row)
        mean_sq = nl.mean(sq, axis=1, keepdims=True)
        rms = nl.power(nl.add(mean_sq, EPS), 0.5)
        normed = nl.divide(row, rms)
        scaled = nl.multiply(normed, w)
        nl.store(result[m*TILE_M:(m+1)*TILE_M, :], scaled)

    return result
```

## Flash-attention-style scaled dot-product attention

Too long to inline here; see `examples/flash_attention.py` in the repo for a worked example. Key tricks:

- Q, K, V tiled along the sequence dimension.
- Online softmax (running max + running sum) to avoid materializing the full attention matrix in SBUF.
- PSUM accumulator for the QK^T and attention*V matmuls.

## When in doubt

Call `nki_skill_lookup` with the operator name and the input shape. The skill DB returns shape-matched recipes from a curated set of NKI patterns.
