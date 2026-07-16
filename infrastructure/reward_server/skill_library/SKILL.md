# NKI (Neuron Kernel Interface) Knowledge Base

Comprehensive reference for writing, optimizing, and debugging NKI kernels
targeting AWS Trainium and Inferentia2 NeuronCores.

---

## 1. NKI Kernel Structure

### 1.1 The `@nki.jit` Decorator

Every NKI kernel is a Python function decorated with `@nki.jit`. This decorator
tells the Neuron compiler to trace and compile the function for NeuronCore
execution.

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def my_kernel(input_tensor, output_tensor):
    # Tile-based computation here
    ...
```

**Key rules:**
- The decorated function receives tensor *handles* (pointers to HBM), not raw data.
- All computation must use NKI language primitives (`nl.*`), not raw Python/NumPy.
- The function body describes a **single NeuronCore's** work; parallelism across
  cores is managed externally.

### 1.2 Tile-Based Programming Model

NKI uses an explicit tiling model. The programmer is responsible for:
1. Declaring tiles that describe regions of data.
2. Loading tiles from HBM into on-chip memory (SBUF/PSUM).
3. Performing computation on tiles.
4. Storing results back to HBM.

```python
@nki.jit
def add_kernel(a, b, out):
    # Process data in tiles of 128 x 512
    TILE_P = 128   # partition dimension -- MUST be 128
    TILE_F = 512   # free dimension -- flexible

    # Loop over tiles
    for p in nl.affine_range(a.shape[0] // TILE_P):
        for f in nl.affine_range(a.shape[1] // TILE_F):
            # Index expressions
            idx_p = p * TILE_P
            idx_f = f * TILE_F

            # Load tiles from HBM -> SBUF
            tile_a = nl.load(a[idx_p:idx_p+TILE_P, idx_f:idx_f+TILE_F])
            tile_b = nl.load(b[idx_p:idx_p+TILE_P, idx_f:idx_f+TILE_F])

            # Compute (runs on Vector Engine)
            tile_out = nl.add(tile_a, tile_b)

            # Store tile SBUF -> HBM
            nl.store(out[idx_p:idx_p+TILE_P, idx_f:idx_f+TILE_F], value=tile_out)
```

### 1.3 Loop Constructs

| Construct | Purpose | Unrolled at compile time? |
|-----------|---------|--------------------------|
| `nl.affine_range(N)` | Tile iteration loops | Yes -- generates N copies |
| `nl.sequential_range(N)` | Sequential iteration (state-carrying) | No -- true loop |
| `nl.static_range(N)` | Alias for affine_range in some versions | Yes |

**When to use which:**
- `affine_range`: default choice for tile loops; enables pipelining.
- `sequential_range`: when a loop iteration depends on the previous one (e.g.,
  accumulation across tiles, scan operations).

---

## 2. Memory Hierarchy

NeuronCore has a multi-level memory hierarchy. Understanding it is critical for
performance.

```
              +-----------+
              |    HBM    |  High Bandwidth Memory (device DRAM)
              | (GB-scale)|  -- All input/output tensors live here
              +-----+-----+
                    |  DMA transfers (nl.load / nl.store)
              +-----+-----+
              |   SBUF    |  State Buffer (on-chip SRAM)
              | (24 MB)   |  -- Primary on-chip working memory
              +-----+-----+
                    |
              +-----+-----+
              |   PSUM    |  Partial Sum Buffer
              | (4 MB)    |  -- Accumulator for matmul results
              +-----+-----+
```

### 2.1 HBM (High Bandwidth Memory)

- **Size:** GBs (varies by instance: 16 GB per core on trn1).
- **Role:** Stores all input and output tensors.
- **Access:** Via DMA through `nl.load()` / `nl.store()`.
- **Bandwidth:** ~200 GB/s per NeuronCore (aggregate).
- **Key concern:** Minimize HBM traffic; reuse data in SBUF/PSUM.

### 2.2 SBUF (State Buffer)

- **Size:** ~24 MB total, organized as 128 partitions x 192 KB each.
- **Role:** Primary on-chip SRAM for operands and intermediate results.
- **Access:** Direct access from Vector Engine and Scalar Engine.
- **Layout:** Data is distributed across 128 partitions (the "partition dimension").
- **Key concern:** Tile partition dimension must be <= 128 to fit.

### 2.3 PSUM (Partial Sum Buffer)

- **Size:** ~4 MB, organized as 128 partitions x 32 KB each.
- **Role:** Accumulator memory used exclusively by the Tensor Engine for
  matrix multiply results.
- **Access:** Tensor Engine writes here; Vector Engine can read.
- **Key concern:** PSUM tiles are the output of `nl.matmul()` before being
  moved to SBUF.

### 2.4 Register Files

- **Scalar registers:** Small register file for the Scalar Engine.
- **GpSimd registers:** Small register file for the GpSimd engine.
- **Usage:** For loop indices, constants, and intermediate scalars.

---

## 3. Tile Constraints

### 3.1 Partition Dimension (P-dim)

- **Fixed at 128.** Every tile's first dimension (partition dimension) must be
  exactly 128 or a divisor of 128 that maps to the 128 SBUF partitions.
- The partition dimension determines how data is distributed across the 128
  SBUF/PSUM partitions.

```python
# CORRECT: partition dim = 128
tile = nl.load(tensor[i:i+128, j:j+512])

# WRONG: partition dim = 256 -- exceeds partition count
tile = nl.load(tensor[i:i+256, j:j+512])   # COMPILE ERROR
```

### 3.2 Free Dimension (F-dim)

- Flexible; limited only by SBUF/PSUM capacity per partition.
- Per partition SBUF capacity: ~192 KB. For float32 (4 bytes), maximum free
  dimension per tile: `192*1024 / 4 = ~49152` elements.
- In practice, keep free dimension to powers of 2 (512, 1024, 2048) for best
  DMA and compute alignment.

### 3.3 Tile Size Guidelines

| Data Type | P-dim | Practical F-dim Range | Notes |
|-----------|-------|-----------------------|-------|
| float32 | 128 | 128 -- 2048 | 4 bytes/element |
| float16 | 128 | 128 -- 4096 | 2 bytes/element |
| bfloat16 | 128 | 128 -- 4096 | 2 bytes/element |
| int8 | 128 | 128 -- 8192 | 1 byte/element |

### 3.4 Alignment Requirements

- Free dimension should be a multiple of 64 for optimal DMA throughput.
- Tensor addresses should be 64-byte aligned.
- Matmul operand free dimensions should be multiples of 128 for Tensor Engine.

---

## 4. Engine Mapping

A NeuronCore contains four compute engines. Each NKI operation maps to a
specific engine.

### 4.1 Tensor Engine

- **Purpose:** Matrix multiplication and convolution.
- **Operations:** `nl.matmul()`
- **Data flow:** Reads from SBUF, writes to PSUM.
- **Throughput:** Peak ~190 TFLOPS (bfloat16/float16 on trn1).
- **Key constraints:**
  - Operands must be in SBUF.
  - Output goes to PSUM (must be copied to SBUF with `nl.copy()` if needed).
  - Contraction dimension must be a multiple of 128.
  - Supports float16, bfloat16, int8, fp8 input types.

```python
# Tile matmul: C[128, N] = A[128, K] @ B[K, N]
a_tile = nl.load(a[p:p+128, :K])   # [128, K] in SBUF
b_tile = nl.load(b[:K, f:f+N])     # [K, N] in SBUF
c_psum = nl.matmul(a_tile, b_tile)  # [128, N] in PSUM
c_sbuf = nl.copy(c_psum, dtype=nl.float32)  # PSUM -> SBUF
nl.store(c[p:p+128, f:f+N], value=c_sbuf)
```

### 4.2 Vector Engine

- **Purpose:** Element-wise operations, reductions, type conversions.
- **Operations:** `nl.add`, `nl.multiply`, `nl.subtract`, `nl.exp`, `nl.log`,
  `nl.rsqrt`, `nl.maximum`, `nl.minimum`, `nl.reduce`, `nl.cast`, etc.
- **Data flow:** Reads from / writes to SBUF.
- **Throughput:** Lower than Tensor Engine but highly versatile.
- **Key strengths:**
  - Operates on SBUF tiles directly.
  - Supports a wide range of unary and binary operations.
  - Can perform reductions along the free dimension.

```python
# Element-wise operations
result = nl.exp(tile_a)                     # exp
result = nl.add(tile_a, tile_b)             # addition
result = nl.multiply(tile_a, tile_b)        # multiplication

# Reductions along free dimension
row_max = nl.max(tile_a, axis=[1])          # [128, 1]
row_sum = nl.add(tile_a, axis=[1])          # [128, 1] -- sum reduction
```

### 4.3 Scalar Engine

- **Purpose:** Control flow, index computation, scalar math.
- **Operations:** Loop indices, conditional masking, address computation.
- **Data flow:** Operates on scalar registers.
- **Usage notes:**
  - Handles loop iteration variables.
  - Computes tile addresses and DMA descriptors.
  - Generally hidden from the programmer (compiler-managed).
  - Avoid putting complex logic here; it becomes a bottleneck.

### 4.4 GpSimd Engine

- **Purpose:** General-purpose SIMD for complex math that cannot run on the
  Vector Engine.
- **Operations:** `nl.gp_simds()` context, trigonometric functions, complex
  branching, custom reductions.
- **Data flow:** Has its own small register file; can access SBUF.
- **Usage notes:**
  - Much slower than Vector Engine -- use as a last resort.
  - Useful for operations like `sin`, `cos`, `atan2`, `pow`.
  - Supports Python-like control flow inside `nl.gp_simds()`.

```python
# Complex math requiring GpSimd
with nl.gp_simds():
    result = nl.sin(tile_a) + nl.cos(tile_b)
```

### 4.5 Engine Pipelining

All four engines can execute **concurrently**. The compiler automatically
pipelines independent operations across engines:

```
Time -->
Tensor:  [matmul tile 0] [matmul tile 1] [matmul tile 2]
Vector:       [add tile 0]    [add tile 1]    [add tile 2]
DMA:     [load tile 1] [load tile 2] [store tile 0] [store tile 1]
```

**Best practice:** Structure kernels so that each engine has work to do
simultaneously. Mix matmul tiles with element-wise post-processing.

---

## 5. Common Patterns

### 5.1 Tiled Matrix Multiplication

**Pattern overview:** Tile over M, N (output dims) with `affine_range`; accumulate
over K (contraction dim) with `sequential_range`. Use `nl.matmul` on TensorE,
accumulate in PSUM, then `nl.copy` to SBUF before storing.

**Key decisions:**
- TILE_M = 128 (partition dim, fixed by hardware)
- TILE_K must be a multiple of 128 (TensorE contraction constraint)
- TILE_N is the free dimension -- tune for SBUF capacity
- Use `nl.zeros(..., buffer=nl.psum)` for the accumulator
- Use `sequential_range` for the K-loop (carries accumulation state)
- Use `affine_range` for the M and N loops (enables pipelining)
- After accumulation, `nl.copy(accum, dtype=...)` moves PSUM -> SBUF before `nl.store`

### 5.2 Softmax

**Pattern overview:** Three-pass algorithm for numerically stable row-wise softmax.
All passes use VectorE reductions and elementwise ops.

**Algorithm (per partition tile):**
1. **Max pass:** Compute row-wise max across the free dimension using `nl.max(..., axis=[1])`.
   When F > TILE_F, accumulate partial maxima with `nl.maximum` across F-tiles.
2. **Exp-sum pass:** Compute `sum(exp(x - row_max))` across the free dimension.
   Subtract max before `nl.exp` for numerical stability. Accumulate partial sums.
3. **Normalize pass:** Compute `exp(x - row_max) / row_sum` and store.

**Key decisions:**
- Tile along the partition dim (TILE_P = 128), iterate over the free dim
- Keep `row_max` and `row_sum` accumulators in SBUF between F-tile iterations
- Use `nl.reciprocal` or division for the final normalization
- For large F, the three-pass approach avoids materializing the full exp tensor in SBUF
- Use `affine_range` for all loops (no cross-iteration data dependency within a pass)

### 5.3 Layer Normalization

**Pattern overview:** Two-pass approach. First pass computes statistics (mean, variance)
via reductions on VectorE. Second pass normalizes, scales, and shifts.

**Algorithm (per partition tile):**
1. **Statistics pass:** Accumulate `sum(x)` and `sum(x^2)` across F-tiles using
   `nl.sum(..., axis=1)`. Compute `mean = sum_x / H` and `var = sum_x2 / H - mean^2`.
2. **Normalize pass:** Compute `inv_std = 1/sqrt(var + eps)`, then
   `(x - mean) * inv_std * gamma + beta` and store.

**Key decisions:**
- Tile along partition dim (TILE_P = 128); the normalization dim H is the free dim
- When H > TILE_F, accumulate partial sums across F-tiles in pass 1
- Load gamma/beta once per F-tile (they broadcast across the partition dim)
- Compute in float32 for numerical stability even if inputs are float16/bfloat16
- Use `nl.rsqrt` or `1.0 / nl.sqrt(...)` for the inverse standard deviation
- RMSNorm variant: skip mean subtraction, only compute `mean(x^2)` in pass 1

### 5.4 Fused Attention (Flash-Attention Style)

**Pattern overview:** Online softmax fused with Q @ K^T and attention-weighted V
accumulation. Combines TensorE (matmul) and VectorE (softmax stats) with
`sequential_range` over the K/V sequence dimension.

**Algorithm (per Q-tile of TILE_M rows):**
1. Load Q tile once, keep in SBUF for reuse across all K/V tiles
2. For each K/V tile (sequential, carries running softmax state):
   a. Compute scores = Q_tile @ K_tile^T on TensorE, copy PSUM -> SBUF
   b. Scale by 1/sqrt(d)
   c. Update online softmax: new_max, rescale running_sum and accumulator
   d. Compute exp(scores - new_max), accumulate into running_sum
   e. Weighted V: acc += exp_scores @ V_tile on TensorE
3. Final normalization: result = acc / running_sum, store

**Key decisions:**
- Outer loop over M-tiles uses `affine_range`; inner loop over N-tiles uses
  `sequential_range` (carries running_max, running_sum, acc state)
- Scores matmul output lands in PSUM; must `nl.copy` to SBUF before VectorE ops
- Online softmax requires rescaling previous accumulators when max updates:
  `exp(old_max - new_max)` correction factor
- V accumulation in PSUM; final result needs `nl.copy` to SBUF before store
- Scale factor `1/sqrt(d)` can be precomputed as a scalar constant

### 5.5 Activation Functions

**General pattern:** Simple elementwise operations with the standard tile loop
(TILE_P = 128, iterate over free dim). Most activations run entirely on VectorE.

**Engine mapping by activation:**

| Activation | Engine | Key ops |
|------------|--------|---------|
| ReLU | VectorE | `nl.maximum(tile, 0.0)` |
| GeLU | VectorE + GpSimd | Polynomial approx on VectorE, `nl.tanh` in `gp_simds()` context |
| SiLU/Swish | VectorE | `nl.exp`, `nl.reciprocal`, `nl.multiply` |
| Sigmoid | VectorE | `nl.exp(neg_x)`, `nl.reciprocal(1 + exp_neg)` |
| Tanh | GpSimd | `nl.tanh` requires `gp_simds()` context |

**Key decisions:**
- GeLU uses the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  The polynomial part (x^3, scaling) runs on VectorE; only `tanh` needs GpSimd.
- SiLU computes sigmoid via `1 / (1 + exp(-x))` using VectorE ops only (no GpSimd needed).
- For fused activations (e.g., matmul + gelu), apply activation in-place on SBUF
  tiles after copying matmul result from PSUM, before the final `nl.store`.
- Use `affine_range` for the tile loop -- all iterations are independent.

---

## 6. Anti-Patterns and Common Mistakes

### 6.1 Wrong Partition Dimension

```python
# BAD: partition dim != 128
tile = nl.load(tensor[i:i+64, j:j+256])    # 64 != 128

# GOOD: partition dim = 128
tile = nl.load(tensor[i:i+128, j:j+256])
```

### 6.2 Exceeding SBUF Capacity

```python
# BAD: tile too large for SBUF (128 * 65536 * 4 bytes = 32 MB > 24 MB)
tile = nl.load(tensor[i:i+128, j:j+65536])

# GOOD: tile fits in SBUF
tile = nl.load(tensor[i:i+128, j:j+2048])
```

### 6.3 Using Python Operators Instead of NKI Primitives

```python
# BAD: Python addition -- not captured by NKI compiler
result = tile_a + tile_b

# GOOD: NKI primitive
result = nl.add(tile_a, tile_b)
```

### 6.4 Incorrect Matmul Dimensions

```python
# BAD: contraction dim not multiple of 128
a = nl.load(A[i:i+128, :50])    # [128, 50] -- 50 not a multiple of 128
b = nl.load(B[:50, j:j+128])
c = nl.matmul(a, b)             # COMPILE ERROR

# GOOD: contraction dim is 128
a = nl.load(A[i:i+128, :128])   # [128, 128]
b = nl.load(B[:128, j:j+128])
c = nl.matmul(a, b)
```

### 6.5 Forgetting PSUM-to-SBUF Copy

```python
# BAD: trying to use PSUM result directly in Vector Engine op
c = nl.matmul(a, b)          # result in PSUM
result = nl.add(c, bias)     # Vector Engine cannot read PSUM directly!

# GOOD: copy to SBUF first
c = nl.matmul(a, b)          # result in PSUM
c_sbuf = nl.copy(c, dtype=nl.float32)  # PSUM -> SBUF
result = nl.add(c_sbuf, bias)
```

### 6.6 Memory Bottleneck: Redundant Loads

```python
# BAD: loading the same data multiple times
for n in nl.affine_range(N_TILES):
    a = nl.load(A[m:m+128, :K])     # loaded every iteration!
    b = nl.load(B[:K, n*128:(n+1)*128])
    nl.matmul(a, b)

# GOOD: hoist invariant loads
a = nl.load(A[m:m+128, :K])         # loaded once
for n in nl.affine_range(N_TILES):
    b = nl.load(B[:K, n*128:(n+1)*128])
    nl.matmul(a, b)
```

### 6.7 Using `sequential_range` When `affine_range` Suffices

```python
# BAD: sequential_range prevents pipelining when iterations are independent
for i in nl.sequential_range(N):
    tile = nl.load(data[i*128:(i+1)*128, :])
    result = nl.exp(tile)
    nl.store(out[i*128:(i+1)*128, :], value=result)

# GOOD: affine_range enables pipelining
for i in nl.affine_range(N):
    tile = nl.load(data[i*128:(i+1)*128, :])
    result = nl.exp(tile)
    nl.store(out[i*128:(i+1)*128, :], value=result)
```

---

## 7. Optimization Tips

### 7.1 Tile Size Selection

- **Start with** `128 x 512` for float32, `128 x 1024` for float16/bfloat16.
- **Scale up** the free dimension if SBUF has room and DMA throughput is the
  bottleneck.
- **Scale down** if you need multiple tiles live simultaneously (e.g., for
  fused operations).
- **Rule of thumb:** `num_live_tiles * tile_size_bytes < 0.8 * SBUF_capacity`.

### 7.2 Engine Pipelining

Structure code so independent operations can overlap:
1. **Load next** tile while computing on the current one.
2. **Store previous** result while loading the next input.
3. **Mix Tensor Engine and Vector Engine** ops for maximum utilization.

Using `affine_range` (not `sequential_range`) enables the compiler to
automatically pipeline iterations.

### 7.3 DMA Prefetching

- The compiler can automatically prefetch DMA transfers when using
  `affine_range`.
- Ensure load/store operations are at the beginning/end of the loop body
  to maximize overlap opportunity.
- For complex access patterns, consider double-buffering manually.

### 7.4 Minimize HBM Round-Trips

- **Fuse** operations: instead of writing intermediate results to HBM and
  reading them back, keep data in SBUF between operations.
- **Example:** fuse matmul + bias + activation into a single kernel instead
  of three separate kernels.

### 7.5 Data Type Selection

- Use **bfloat16/float16** for Tensor Engine operations when precision allows.
  Throughput is 2x compared to float32.
- Use **float32** for accumulation and numerically sensitive operations
  (reductions, normalization).
- Tensor Engine supports mixed precision: float16 inputs, float32
  accumulation.

### 7.6 Reduction Optimization

- Row-wise reductions (along free dimension) are fast on the Vector Engine.
- Column-wise reductions (along partition dimension) are expensive because
  they require cross-partition communication.
- Restructure algorithms to reduce along the free dimension whenever possible.

### 7.7 Matmul Optimization

- Keep contraction dimension as large as possible (multiples of 128).
- Use `sequential_range` for the K-reduction loop in large matmuls.
- Accumulate in PSUM (float32) then convert to target dtype before storing.
- For small matmuls, consider batching multiple small matmuls into one
  larger tile operation.

### 7.8 Memory Layout Considerations

- Data in HBM should be laid out so that the partition dimension (dim 0)
  is contiguous in memory.
- Transposing data in HBM is expensive; transpose in SBUF when possible
  using `nl.transpose()`.
- Align tensor base addresses to 64 bytes.

### 7.9 Profiling-Driven Optimization

Use `neuron-profile` to identify bottlenecks:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Low Tensor Engine util | Not enough matmul work | Increase tile sizes, batch matmuls |
| Low Vector Engine util | Waiting on DMA or Tensor Engine | Fuse more element-wise ops |
| High DMA time | Memory-bound | Increase tile reuse, reduce HBM traffic |
| High Scalar Engine time | Complex control flow | Simplify addressing, use affine_range |
| High GpSimd time | Complex math ops | Replace with Vector Engine approximations |

### 7.10 Numerical Stability

- Always subtract the max before `exp()` in softmax-like operations.
- Use float32 for accumulation even when inputs are float16.
- Be aware that `rsqrt` has limited precision for very small values.
- For layer norm / RMS norm, compute variance in float32.

---

## 8. Debugging Checklist

When a kernel fails to compile or produces incorrect results:

1. **Check partition dimension:** Must be exactly 128 (or valid sub-partition).
2. **Check SBUF capacity:** Total live tiles must fit in 24 MB.
3. **Check PSUM capacity:** Matmul accumulator tiles must fit in 4 MB.
4. **Check contraction dimension:** Must be a multiple of 128 for `nl.matmul`.
5. **Check data types:** Tensor Engine only supports float16/bfloat16/int8/fp8.
6. **Check PSUM copies:** Always `nl.copy()` PSUM tiles to SBUF before using
   them in Vector Engine operations.
7. **Check loop constructs:** Use `sequential_range` only when iterations
   have data dependencies.
8. **Check tensor shapes:** Ensure input/output shapes match expected tile
   dimensions.
9. **Run with `NKI_LOG_LEVEL=DEBUG`** for detailed compiler diagnostics.
10. **Compare against NumPy/PyTorch** reference implementation with identical
    inputs to isolate numerical issues.

---

## 9. Quick Reference: NKI Language Primitives

### Data Movement
| Primitive | Description |
|-----------|-------------|
| `nl.load(tensor_slice)` | HBM -> SBUF |
| `nl.store(tensor_slice, value=tile)` | SBUF -> HBM |
| `nl.copy(tile, dtype=...)` | PSUM -> SBUF or dtype conversion in SBUF |
| `nl.transpose(tile)` | Transpose tile in SBUF |

### Arithmetic (Vector Engine)
| Primitive | Description |
|-----------|-------------|
| `nl.add(a, b)` | Element-wise addition |
| `nl.subtract(a, b)` | Element-wise subtraction |
| `nl.multiply(a, b)` | Element-wise multiplication |
| `nl.divide(a, b)` | Element-wise division |
| `nl.exp(a)` | Element-wise exponential |
| `nl.log(a)` | Element-wise natural log |
| `nl.rsqrt(a)` | Element-wise reciprocal sqrt |
| `nl.reciprocal(a)` | Element-wise 1/x |
| `nl.maximum(a, b)` | Element-wise max |
| `nl.minimum(a, b)` | Element-wise min |
| `nl.abs(a)` | Element-wise absolute value |
| `nl.negative(a)` | Element-wise negation |

### Reductions (Vector Engine)
| Primitive | Description |
|-----------|-------------|
| `nl.max(a, axis=[1])` | Max along free dimension |
| `nl.min(a, axis=[1])` | Min along free dimension |
| `nl.add(a, axis=[1])` | Sum along free dimension |

### Matrix Operations (Tensor Engine)
| Primitive | Description |
|-----------|-------------|
| `nl.matmul(a, b)` | Matrix multiply, result in PSUM |

### Memory Allocation
| Primitive | Description |
|-----------|-------------|
| `nl.zeros(shape, dtype, buffer)` | Allocate zero-filled tile |
| `nl.full(shape, value, dtype)` | Allocate constant-filled tile |

### Control Flow
| Primitive | Description |
|-----------|-------------|
| `nl.affine_range(N)` | Unrolled loop (enables pipelining) |
| `nl.sequential_range(N)` | Sequential loop (for dependencies) |
| `nl.gp_simds()` | Context for GpSimd engine operations |

## 10. neuronxcc 2.23 quick reference (verified on this reward server)

This section is grounded in kernels that actually compiled and verified on the
deployed `neuronxcc 2.23` Neuron SDK. Use the `nl.*` high-level API below — it is
confirmed present. Do NOT use the `nisa.dma_copy` / `nisa.activation(dst=...)`
style from newer NKI 0.4.0 guides; that API is not available here.

### 10.1 Pre-flight checklist (violating any of these fails compilation/verification)
1. `import neuronxcc.nki as nki` and `import neuronxcc.nki.language as nl` — never `import nki`.
2. Function is named `nki_kernel` and decorated `@nki.jit`.
3. READ the input rank first. Inputs are often 3-D `(B, M, N)` or 4-D `(N, C, H, W)`, not just 2-D.
4. Allocate the FULL output: `result = nl.ndarray(<full shape>, dtype=x.dtype, buffer=nl.shared_hbm)`.
5. The partition axis of every TILE is 128. Tile the leading/collapsed axis by 128.
6. EVERY output element must be written by an `nl.store`. Iterate ALL leading dims.
7. Reductions run over the FREE axis: `nl.sum(tile, axis=1)` on a `(128, F)` tile.
8. `return result`.

### 10.2 Minimal working kernel (2-D), verified idiom
```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(M // TILE_M):
        row = nl.load(x[i * TILE_M:(i + 1) * TILE_M, :])   # (128, N)
        out = nl.relu(row)                                  # compute
        nl.store(result[i * TILE_M:(i + 1) * TILE_M, :], out)
    return result
```

### 10.3 Higher-rank template — loop leading dims, tile by 128 (verified idiom)
```python
# 3-D (B, M, N): loop batch, tile M by 128
@nki.jit
def nki_kernel(x):
    B, M, N = x.shape
    TILE_M = 128
    result = nl.ndarray((B, M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for b in nl.affine_range(B):
        for i in nl.affine_range(M // TILE_M):
            t = nl.load(x[b, i * TILE_M:(i + 1) * TILE_M, :])
            nl.store(result[b, i * TILE_M:(i + 1) * TILE_M, :], nl.relu(t))
    return result
```

### 10.4 PyTorch → neuronxcc 2.23 `nl` op table (verified present)
| PyTorch | 2.23 NKI (`nl`) |
|---|---|
| `a + b` / `a - b` / `a * b` / `a / b` | `nl.add` / `nl.subtract` / `nl.multiply` / `nl.divide` |
| `a * scalar` | `nl.multiply(a, scalar)` |
| `torch.exp/log/sqrt/rsqrt` | `nl.exp` / `nl.log` / `nl.sqrt` / `nl.rsqrt` |
| `1/x` | `nl.reciprocal(x)` |
| `torch.relu/sigmoid/tanh/gelu/erf` | `nl.relu` / `nl.sigmoid` / `nl.tanh` / `nl.gelu` / `nl.erf` (all EXIST — call them) |
| `torch.maximum/minimum(a,b)` | `nl.maximum` / `nl.minimum` |
| `torch.sum/max/min(x, dim=1)` | `nl.sum(x, axis=1)` / `nl.max(...)` / `nl.min(...)` (FREE axis only) |
| `torch.mean(x, dim=1)` | `nl.sum(x, axis=1)` then `nl.multiply(s, 1.0/F)` |
| `a @ b` | `nl.matmul(a, b)` |
| `x.T` | `nl.transpose(x)` |

### 10.5 Anti-patterns that cause failures here
- `import nki` (wrong) instead of `import neuronxcc.nki as nki`.
- `nl.pow` / `nl.exp2` — do NOT exist. Use `nl.exp(b * nl.log(a))`.
- Unpacking `M, N = x.shape` on a 3-D/4-D input — read the real rank.
- Reducing over the partition axis — reductions must be over the free axis (`axis=1`).
- Storing only one tile slice of a multi-tile / multi-batch output — leaves regions
  unwritten → "Output ... has no store def" or wrong numbers. Iterate every leading dim.
- Tiling by a size that is not 128 on the partition axis.

## 11. VERIFIED recipes (copy these exact 2.23 idioms — proven numerically correct on Trn1)

These are complete kernels that PASSED on-device verification (np.allclose,
atol=rtol=1e-3). When a task matches, adapt one of these rather than inventing
structure — most numerical (verify) failures come from getting softmax/reduction
details subtly wrong. Reductions are over the FREE axis (`axis=1`) on a `(128, F)` tile.

### 11.1 Row-wise softmax — VERIFIED (max_abs_error 3e-7)
```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(M // TILE_M):
        row = nl.load(x[i*TILE_M:(i+1)*TILE_M, :])   # (128, N)
        rmax = nl.max(row, axis=1)                    # (128, 1) — numerical stability
        shifted = nl.subtract(row, rmax)
        e = nl.exp(shifted)
        s = nl.sum(e, axis=1)                         # (128, 1)
        out = nl.divide(e, s)
        nl.store(result[i*TILE_M:(i+1)*TILE_M, :], out)
    return result
```
Softmax MUST subtract the row max before `nl.exp` (stability) and divide by the
row sum. Both reductions are `axis=1`. This is the single most common L2 numeric bug.

### 11.2 Scaled dot-product attention — structure that VERIFIED on B,H,S,D inputs
```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_kernel(q, k, v):                # shapes (B, H, S, D)
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    result = nl.ndarray((B, H, S, D), dtype=q.dtype, buffer=nl.shared_hbm)
    for b in nl.affine_range(B):
        for h in nl.affine_range(H):
            qbh = nl.load(q[b, h])                    # (S, D)
            kbh = nl.load(k[b, h])
            vbh = nl.load(v[b, h])
            scores = nl.multiply(nl.matmul(qbh, nl.transpose(kbh)), scale)  # (S, S)
            # row-softmax over the last axis (axis=1): subtract max, exp, divide by sum
            smax = nl.max(scores, axis=1)
            e = nl.exp(nl.subtract(scores, smax))
            probs = nl.divide(e, nl.sum(e, axis=1))
            out = nl.matmul(probs, vbh)               # (S, D)
            nl.store(result[b, h], out)
    return result
```
Key numeric points that make attention VERIFY (not just compile):
- Scale AFTER the QK^T matmul by `1/sqrt(D)`.
- Softmax the score rows with the max-subtraction trick (see 11.1).
- For GQA/MHA where K/H heads differ (`Hk < H`), map query head h to kv head
  `h // (H // Hk)` when loading k/v.
- Apply an additive mask (`scores + mask`) BEFORE the softmax, not after.

### 11.3 RMSNorm — VERIFIED (max_abs_error 2e-4)
```python
@nki.jit
def nki_kernel(x):
    M, N = x.shape
    TILE_M = 128
    eps = 1e-6
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(M // TILE_M):
        row = nl.load(x[i*TILE_M:(i+1)*TILE_M, :])
        variance = nl.multiply(nl.sum(nl.multiply(row, row), axis=1), 1.0 / N)
        normed = nl.multiply(row, nl.rsqrt(nl.add(variance, eps)))
        nl.store(result[i*TILE_M:(i+1)*TILE_M, :], normed)
    return result
```

### 11.4 Matmul (C = A @ B) — VERIFIED (max_abs_error 2.5e-5)
On the 2.23 high-level API, `nl.matmul` handles K-accumulation internally — you do
NOT need to hand-manage PSUM for a plain matmul. Tile the M (partition) axis by 128.
```python
@nki.jit
def nki_kernel(a, b):                 # a:(M,K)  b:(K,N)
    M, K = a.shape
    _, N = b.shape
    TILE_M = 128
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(M // TILE_M):
        at = nl.load(a[i*TILE_M:(i+1)*TILE_M, :])   # (128, K)
        bt = nl.load(b)                              # (K, N)
        nl.store(result[i*TILE_M:(i+1)*TILE_M, :], nl.matmul(at, bt))
    return result
```
For a FUSED matmul (e.g. matmul+bias+relu), compute `nl.matmul(at, bt)` then apply
`nl.add(..., bias)` / `nl.relu(...)` on the (128, N) tile before the single nl.store.
If you drop to the low-level `nisa.nc_matmul` (rarely needed here), THEN the PSUM
rules apply: one nl.zeros(...,buffer=nl.psum) region accumulated across the K-loop,
copy PSUM->SBUF before store, never PSUM->HBM. Prefer `nl.matmul` unless a task needs it.
