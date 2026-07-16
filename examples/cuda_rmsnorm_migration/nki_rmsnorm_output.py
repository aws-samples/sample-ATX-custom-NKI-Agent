import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    eps = 1e-6
    out_shape = (M, N)
    result = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(M // TILE_M):
        row = nl.load(x[i * TILE_M:(i + 1) * TILE_M, :])
        sq = nl.multiply(row, row)
        ssum = nl.sum(sq, axis=1)
        variance = nl.multiply(ssum, 1.0 / N)
        rms = nl.rsqrt(nl.add(variance, eps))
        normed = nl.multiply(row, rms)
        nl.store(result[i * TILE_M:(i + 1) * TILE_M, :], normed)
    return result
