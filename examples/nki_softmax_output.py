"""Example output: the agent's NKI rewrite of triton_softmax_input.py.

Numerically stable row-wise softmax. TILE_M = 128 partition dim,
full row in SBUF (assumes N fits in scratch).
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128

    result = nl.ndarray(shape=(M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        row = nl.load(x[m * TILE_M : (m + 1) * TILE_M, :])
        row_max = nl.max(row, axis=1, keepdims=True)
        shifted = nl.subtract(row, row_max)
        exp = nl.exp(shifted)
        denom = nl.sum(exp, axis=1, keepdims=True)
        out = nl.divide(exp, denom)
        nl.store(result[m * TILE_M : (m + 1) * TILE_M, :], out)

    return result
