# GPU Structure-Function Histogram Kernels: Theory, Correctness, and Benchmark Log

**Status:** Research / prototype phase (`gpu/GPUPrototypeKernels.jl`). Production code remains in `ext/StructureFunctionsGPUExt.jl` (global `(N,N)` launch with Float32 count atomics).

**Primary benchmark:** `N = 20\,000`, seed `42`, `Float32`, `L2SFType`, linear bins `range(0.1, 2.0; length=21)` (20 interior bins, `NB = 20`), 2D fields `(x, u) ∈ ℝ²`, NVIDIA A100-SXM4-80GB, Julia `--project=gpu`, single-threaded host.

This document records **everything learned** during the prototype campaign: mathematical problem statement, algorithm families, floating-point and atomic semantics, pair/tile indexing, CUDA compiler constraints, literature connections, measured timings, parity policy, and open issues. It is the hand-written counterpart to the auto-generated `GPU_timings_and_theory.md` from `gpu_full_benchmark.jl`.

---

## 1. Problem statement

### 1.1 Unordered pair histogram

Given `N` points with positions `x_i ∈ ℝ^{N_dims}` and field values `u_i ∈ ℝ^{N_dims}` (prototype tests use `N_dims = 2`; device kernels pad to 3D), consider all **unordered distinct pairs**

\[
\mathcal{P} = \{(i,j) : 1 \le i < j \le N\}, \qquad |\mathcal{P}| = \frac{N(N-1)}{2}.
\]

For each pair `(i,j)`:

1. **Separation:** `d_{ij} = \|x_j - x_i\|` (Euclidean; 2D or 3D depending on kernel variant).
2. **Bin index:** `b_{ij} = \mathrm{digitize}(d_{ij}; \text{edges})` using the same linear-bin rules as CPU `LinearBinEdges`.
3. **Structure-function sample:** With unit separation vector `r̂_{ij} = (x_j - x_i) / d_{ij}` and increment `Δu_{ij} = u_j - u_i`,
   \[
   v_{ij} = \mathrm{SF}(Δu_{ij}, \hat{r}_{ij}).
   \]
   For `L2SFType` (longitudinal/transverse L² structure function), this is the standard projected second-order kernel implemented in `StructureFunctionTypes.jl`.

4. **Histogram accumulation** into `NB = n_{\text{bins}} - 1` interior bins:
   \[
   S[b] \mathrel{+}= v_{ij}, \qquad C[b] \mathrel{+}= 1 \quad \text{when } 1 \le b \le NB.
   \]

Pairs with `d_{ij}` below the first edge or above the last edge contribute to **guard bins** (index `0` or `n_{\text{bins}}`) and are **not** accumulated into `S`, `C` — matching CPU gold.

### 1.2 Work scale at `N = 20\,000`

| Quantity | Value |
|----------|------:|
| `N` | 20,000 |
| `\|𝒫\|` | 199,990,000 ≈ **200M pairs** |
| Interior bin count `NB` | 20 |
| CPU gold `Σ C[b]` (in-bin only) | **194,251,328** |
| Pairs outside interior bins | ≈ 5.74M |

The output is a **small histogram** (20 sums + 20 counts), not an `N × N` matrix. All algorithmic effort is in **how** the 200M pair updates are scheduled and accumulated on GPU.

### 1.3 Linear digitization (device)

Uniform edges `e_0, e_1, …, e_{NB}` with step `Δe = e_1 - e_0`. Precomputed on host:

\[
\text{inv\_step} = \frac{1}{Δe}, \qquad \text{offset} = 1 - \frac{e_0}{Δe}.
\]

For distance `x`, interior bin (1-based) uses FMA-style mapping with integer repair (see `BinEdges_timings_and_theory.md` on CPU). Device prototype uses `_gpu_digitize_linear` equivalent to CPU.

**`nosqrt` variant:** Compare `d²` to squared edges `e_k²` via `_gpu_digitize_linear_sq` — mathematically equivalent for linear bins on exact arithmetic; still requires `sqrt(d²)` (or equivalent) for `r̂`. Benchmark showed **nosqrt slower** on GPU (more edge comparisons in the backward scan), not a win at `N=20k`.

---

## 2. Production baseline (`StructureFunctionsGPUExt.jl`)

### 2.1 Launch geometry

Production launches `ndrange = (N, N)` with one thread per `(i,j)`. Valid pairs satisfy `i < j`. Each thread:

- Loads `x_i, x_j, u_i, u_j` from global memory (random access pattern).
- Computes `d`, `b`, `v`.
- **`@atomic` updates global `output[b]` and `counts[b]`** (Float32 for both today).

### 2.2 Why it is slow (~0.097 s kernel at `N=20k`, prototype mirror)

Per valid pair, **two global atomics** (sum + count) contend on ~20 bin slots:

\[
\text{global atomics per pair} \approx 2, \qquad \text{total} \approx 2|\mathcal{P}| \approx 4 \times 10^8.
\]

Even with hardware atomic throughput, **serialization on hot bins** (many pairs land in similar separation bins for random points in a bounded domain) limits occupancy. Measured kernel time ~**0.096–0.097 s** on A100 for prototype mirrors of this path.

End-to-end production call (`gpu_calculate_structure_function` with host staging each invocation) measured **~0.144 s** in the same session — staging and allocation dominate relative to the prototype micro-benchmark that amortizes setup differently.

### 2.3 Why it is wrong at large `N` (Float32 counts)

**Counts stored as Float32 atomics:** integer count `C[b]` is updated via `@atomic counts[b] += 1.0f0`.

IEEE-754 `Float32` has **24-bit significand**. Integer values above `2^{24} = 16{,}777{,}216` are not representable exactly. Hot bins with tens of millions of `+1` updates suffer **ULP gaps**: many atomic adds are **no-ops** at the representable grid.

Measured at `N=20k`, seed 42:

| Path | `Σ counts` | `Δ` vs gold |
|------|----------:|------------:|
| CPU gold (Int64 / exact) | 194,251,328 | 0 |
| Float32 serial CPU gold | 151,928,432 | **−42,322,896** |
| Global GPU Float32 (`baseline_grid_N2`) | ≈ 151.9M | **−42,322,904** |
| Global GPU **UInt32** (`baseline_linear_global_u32`) | 194,251,328 | **0** |

**Fix for counts:** `UInt32` (or `Int32`) device atomics — same algorithm, exact total in-bin counts.

**Sums remain Float32** in prototypes; parity vs truth uses **Float64 serial reference**, not Float32 serial gold (see §4).

---

## 3. Reference hierarchy and parity policy

### 3.1 What “gold” means

| Reference | Role |
|-----------|------|
| **`cpu_gold_histogram`** | Int64 counts + Float32 sums; serial pair loop; `_pair_from_linear`; 3D padding for 2D input |
| **`cpu_f64_serial_sums`** | Float64 sums, same pair order — **sum parity truth** |
| **`validate_f64_references`** | Proves f64 serial ≈ f64 private ≈ f64 stride ≈ f64 blockshared; BigFloat spot check |
| **Float32 serial sums** | Diagnostic only — **~5% low** on hot bins at `N=20k`; must **not** be used as GPU sum reference |

Validated at `N=20k` (from `prove_f32_accumulation.jl` / benchmark header):

- `max|serial − private_all_f64|` ≈ **1.3×10⁻⁴**
- BigFloat vs f64 serial on bin 5: **Δ ≈ 8.85×10⁻⁸**
- Bin 5 f64 sum ≈ **4.36645×10⁶**

### 3.2 Benchmark acceptance (`benchmark_helpers.jl`)

For each variant:

- **`Δcnt = Σ counts − gold.n_in`** — require **0** for u32 paths.
- **`max|Δsum|`** vs **f64 serial** with `rtol=1e-4`, **`atol=500`**.
- **Strict `ok`:** additionally **`counts_bin_exact`** — per-bin `C[b]` matches gold Int64 vector exactly.

Observed: many fast u32 paths have **`Δcnt=0`** and **`sums_ok=true`** but fail strict parity because **`max_Δcnt_bin = 3`** — some bin differs by 3 counts while totals cancel (§7.3). Label: `FAIL(cnt Δ=+0 max_bin=3)`.

---

## 4. Float32 sum accumulation: order and atomics

### 4.1 Non-associativity

Float32 addition is not associative. Different **pair processing orders** yield different roundoff:

\[
((a + b) + c) \neq (a + (b + c)) \quad \text{in general}.
\]

For **signed** structure-function samples (L2 kernel can produce mixed signs), partial sums can be large with opposing terms — **catastrophic cancellation** amplifies order dependence.

### 4.2 CPU simulation results (`prove_f32_accumulation.jl`, `N=20k`)

Same pair set, same digitization; only accumulation schedule differs:

| Schedule | Sum error vs f64 serial (typical) | Notes |
|----------|-----------------------------------|-------|
| **Serial f32** | **~O(10⁵)** on total / hot bins | Not a valid reference |
| **Grid-stride global f32 atomics** | ~**212k** low on sums; **~47% no-op** `+=` in hot bin | Matches production class |
| **Private / blockshared f32** | **~10–12** per bin vs f64 twin | Same schedule in f64 → agreement |
| **f64 serial / private / blockshared** | **≤ 1.3×10⁻⁴** mutual | Validated reference class |

**Conclusion:** GPU sum parity must be judged against **f64 serial**, not f32 serial gold. Float32 **private/blockshared/tiled** paths are acceptable if `max|Δsum| ≪ 500` atol.

### 4.3 Why global Float32 sums fail even with u32 counts

`baseline_linear_global_u32`: **exact counts**, but `max|Δsum| ≈ 2.12×10⁵` vs f64 — same catastrophic global atomic order as Float32-count global path. **Do not ship** for sums despite fixed counts.

---

## 5. Algorithm families (prototype kernels)

All optimized prototypes abandon the `(N,N)` grid for a **linear pair index** `k ∈ {1,…,|\mathcal{P}|}` mapped to `(i,j)` via `_pair_from_linear` (§6).

### 5.1 Global atomic, grid-stride pairs (`baseline_linear_*`)

- **Workers:** `W = min(262144, |\mathcal{P}|)`.
- Thread `w` processes `k = w, w+W, w+2W, …`.
- **Atomics:** global `output`, `counts` every pair.
- **Time:** ~**0.096 s** (same as `(N,N)` baseline).
- **Counts:** u32 fixes total; f32 loses ~42M.
- **Sums:** large error vs f64 for global path.

### 5.2 Register-private histogram per worker (`private_*`)

Each global worker maintains **`MVector{NB}`** partial histogram in registers (CUDA: `MVector{64,FT}(undef)` + zero loop works in **simple** kernels).

After processing its pair subsequence, **one global atomic flush per bin**.

- **Pros:** Few global atomics per worker (`≈ 2·NB` per worker).
- **Cons:** At `W=256k`, register pressure; **slower than blockshared** at full worker count (~**0.021 s** vs **0.012 s**). Float32 **counts** still drift (+508 at 256k workers).

### 5.3 Block-local shared histogram (`blockshared_*`)

**Canonical intermediate fast path** before tiling.

Per CUDA block (workgroup size `ws = 256`):

1. **`@localmem`** arrays `shared_sums[1:NB]`, `shared_cnts[1:NB]`.
2. Grid-stride over pair indices assigned to this block.
3. **`@atomic` updates into shared memory** (block-local contention only).
4. Block flush: **`@atomic` global `+=`** once per bin.

**Global atomics reduced** from `O(|\mathcal{P}|)` to `O(n_{\text{blocks}} \cdot NB)`:

\[
n_{\text{blocks}} = \left\lceil \frac{W}{ws} \right\rceil, \quad W = 262144 \Rightarrow n_{\text{blocks}} = 1024.
\]

Flush volume ≈ **1024 × 20 = 20{,}480** global atomics per kernel launch (vs 400M).

**Measured (`blockshared_256k_w256_u32`):** kernel **~0.0115 s**; `Δcnt=0`; `max|Δsum| ≈ 10` vs f64.

**Register-private block variant (`blockshared_regpriv_u32`):** per-thread `MVector` partials merged once into shared — **slower** (~**0.020 s**) on A100; not beneficial here.

### 5.4 CADISHI-style tiled histogram (`tiled{64,128}_*`)

**Literature:** Particle–particle short-range lists on GPUs routinely **tile** points into shared memory so threads reuse loaded coordinates (CADISHI: [Shimizu et al., Comput. Phys. Commun. 2018](https://doi.org/10.1016/j.cpc.2018.10.018); similar 2-body tile ideas in [Pitaksirianan et al., DAPD 2019](https://cse.usf.edu/~tuy/pub/DAPD19.pdf)).

**Idea:** Partition indices into tiles of `T` points (`T ∈ {64, 128}`). Upper-triangle **tile pairs** `(t_i, t_j)` with `t_i ≤ t_j`:

\[
n_{\text{tiles}} = \left\lceil \frac{N}{T} \right\rceil, \qquad n_{\text{tile blocks}} = \frac{n_{\text{tiles}}(n_{\text{tiles}}+1)}{2}.
\]

At `N=20k`:

| `T` | `n_tiles` | `n_tile_blocks` |
|----:|----------:|----------------:|
| 64 | 313 | **49,141** |
| 128 | 157 | **12,403** |

One CUDA block per tile-pair `(t_i, t_j)`:

1. **Cooperative load** of tile `t_i` points into `shared_xi`, `shared_ui` (and tile `t_j` if `t_i < t_j`).
2. **Inner loops** over pairs **within tiles** using **direct indices** `(i_a, j_b)` — **no binary search** in the hot loop.
3. Block-local histogram in shared mem (same as blockshared).
4. Global flush once per bin per block.

**Memory traffic win:** each point coordinate loaded from global memory **O(n_tiles)** times across the tile-row/column schedule instead of **O(N)** random loads per point in unstructured pair loops.

**Measured winner (`tiled128_2d_u32_w256`):** kernel **~0.0045 s** (~**21×** vs global, ~**2.5×** vs blockshared u32).

#### 5.4.1 Diagonal vs off-diagonal tile pairs

- **`t_i < t_j`:** all `n_i × n_j` cross pairs (both tiles loaded).
- **`t_i = t_j`:** upper triangle **within** partial tile of size `n_i`, count `n_i(n_i-1)/2`.

#### 5.4.2 `2d` vs 3D-padded distance

CPU gold pads `(x,y)` to `(x,y,0)` and uses 3D distance. **`2d` kernels** use

\[
d_{ij} = \sqrt{(x_j-x_i)^2 + (y_j-y_i)^2}, \qquad \hat{r} \in \mathbb{R}^2.
\]

For true 2D data this is physically correct; sums differ slightly from 3D-padded gold (`max|Δsum|` ~**9.5** vs ~**19.5** for 3D tiled64 at `N=20k`). **Use 2D path in production when `N_dims==2`.**

#### 5.4.3 `regpriv` on tiled kernels (CUDA constraint)

**Intended:** per-thread register partial histogram before merging to shared (fewer shared atomics).

**Observed on CUDA:** `SA.MVector{64,FT}(undef)` inside **tile** kernels triggers **`gpu_gc_pool_alloc`** (heap) or register spill — **InvalidIRError**. Same `MVector` works in **simpler** kernels (`private`, `blockshared_regpriv`) with lower register/shared footprint.

**Workaround in prototype:** tiled `regpriv` variants compile to **shared atomic path identical to non-regpriv** (honest benchmark label; timings match within noise).

**Failed alternatives documented:**

| Approach | Result |
|----------|--------|
| `MVector(undef)` in `@eval` kernel | `gpu_gc_pool_alloc` |
| `ntuple` zero-init `MVector` | still heap |
| `@localmem` per-thread slice `256×64` | **131072 B smem** > **48 KB** default block limit |
| Parse-time `@macro` + `MVector` | still heap in tile kernel (register pressure) |

#### 5.4.4 `nosqrt` digitization

Bin from `d²` vs edges `e_k²`. No sqrt in digitize; still need `sqrt` for `r̂`. **Slower** (~**0.008–0.010 s**) — backward edge scan dominates; not used.

---

## 6. Pair indexing mathematics

### 6.1 Row-major upper triangle

Index pairs by row `i = 1,…,N-1`, column `j = i+1,…,N`. Row `i` contains `N-i` pairs. Cumulative pairs before row `i`:

\[
\mathrm{row\_start}(i) = \sum_{r=1}^{i-1} (N-r) = (i-1)N - \frac{(i-1)i}{2}.
\]

Linear index `k ∈ {1,…,|\mathcal{P}|}` maps to `(i,j)` by **integer binary search** on `i` using `row_start` — **O(log N)** comparisons, all integer, GPU-safe.

**Do not** use closed-form `i ≈ N - sqrt(...)` in Float32 on GPU for `N=20k`: proved to **miss ~150k pairs** in earlier diagnosis.

### 6.2 Tile index

Same combinatorics on **tile indices** `t_i, t_j ∈ {1,…,n_{\text{tiles}}}` with `_tile_from_linear`.

---

## 7. Full benchmark table (`benchmark_prototypes.jl`, A100, `N=20k`)

Kernel times below; staging adds ~0.0002–0.002 s unless noted. **Timing baseline:** `baseline_linear_global` ≈ **0.0962 s**.

| Variant | Kernel (s) | vs ~0.096s | cnt | Δcnt | max\|Δsum\| f64 | Notes |
|---------|----------:|-----------:|:---:|:----:|----------------:|-------|
| baseline_grid_N2 | 0.0963 | 1.0× | f32 | −42.3M | 2.13e5 | production-like |
| baseline_linear_global | 0.0962 | 1.0× | f32 | −42.3M | 2.12e5 | timing ref |
| baseline_linear_global_u32 | 0.0962 | 1.0× | u32 | 0 | 2.12e5 | exact Σcnt, bad sums |
| private_4k | 0.1205 | 0.8× | f32 | −70 | 5.3 | too few workers |
| private_32k | 0.0212 | 4.4× | f32 | +238 | 16.8 | |
| private_256k | 0.0208 | 4.5× | f32 | +508 | 40 | |
| blockshared_32k | 0.0254 | 3.7× | f32 | +10 | 179 | |
| blockshared_256k | 0.0122 | 7.6× | f32 | +22 | 10.5 | |
| **blockshared_256k_u32** | **0.0115** | **8.2×** | u32 | 0 | **10** | parity ref |
| blockshared_regpriv_u32 | 0.0199 | 4.8× | u32 | 0 | 3.3 | regpriv slower |
| **tiled64_u32_w256** | **0.0064** | **15×** | u32 | 0 | 19.5 | |
| tiled64_2d_u32 | 0.0059 | 16× | u32 | 0 | 21 | |
| tiled64_nosqrt_u32 | 0.0095 | 10× | u32 | 0 | 20 | slower |
| tiled64_regpriv_u32 | 0.0064 | 14× | u32 | 0 | 20.5 | = base (fallback) |
| **tiled128_u32_w256** | **0.0050** | **18×** | u32 | 0 | 9.5 | |
| **tiled128_2d_u32** | **0.0045** | **21×** | u32 | 0 | **9.5** | **fastest** |
| tiled128_nosqrt_u32 | 0.0079 | 12× | u32 | 0 | 10 | |
| tiled128_regpriv_* | 0.0045–0.0078 | — | u32 | 0 | ~10 | = non-regpriv |
| tiled64_w128_u32 | 0.0052 | 17× | u32 | 0 | 20 | ws=128 |
| **production ext** | **0.1444** | — | f32 | broken | — | incl. staging |

**Strict script label:** `0 / 28` pass strict `ok` because of **`max_Δcnt_bin = 3`** on most u32 fast paths while **`Δcnt=0`** (§7.3).

### 7.1 Speed ladder (kernel only)

\[
\underbrace{0.096}_{\text{global}} \;\to\; \underbrace{0.012}_{\text{blockshared u32}} \;\to\; \underbrace{0.0064}_{\text{tiled64 u32}} \;\to\; \underbrace{0.0045}_{\text{tiled128 2d u32}} \;\approx\; 21\times \text{ speedup}.
\]

Production end-to-end **0.144 s** vs **~0.0045 s** kernel ⇒ **~32×** kernel gap (staging not optimized in ext).

### 7.2 What did **not** help

- **Global UInt32** — fixes counts, not speed.
- **Register-private (regpriv)** — slower on blockshared and infeasible on tiled (CUDA).
- **nosqrt digitize** — slower than sqrt+linear FMA path at these bin counts.
- **Float32 counts** anywhere at `N=20k`.

### 7.3 Per-bin count mismatch (`max_Δcnt_bin = 3`)

Several u32 kernels: **`Σ_b C[b]` exact** but **`∃ b : |C[b] - C^{\text{gold}}[b]| = 3`**. Errors in different bins cancel in the sum. Likely causes under investigation:

- Tile boundary pair enumeration edge case (partial tiles at `i_0 + n_i - 1`).
- Off-by-one in within-tile upper triangle when `n_i < T`.
- Not observed in f64 CPU tiled simulator yet (simulator TODO).

**Action before production:** add `cpu_tiled_gold` parity test; print bin index achieving max Δ.

---

## 8. Launch configuration reference

| Parameter | Typical value | Meaning |
|-----------|---------------|---------|
| `W` | `min(262144, \|𝒫\|)` | grid-stride workers (blockshared/private) |
| `ws` | 256 (128 for one tiled64 config) | threads per block |
| `n_blocks` | `⌈W/ws⌉` | CUDA blocks for pair stride |
| `n_tile_blocks` | `n_tiles(n_tiles+1)/2` | CUDA blocks for tiled kernels |
| `ndrange` (tiled) | `n_tile_blocks × ws` | one block group per tile pair |

Host staging (`_stage_host_to_device`): pads 2D→3×N, allocates device buffers, copies once per prototype call unless `device_resident` buffers reused.

---

## 9. Production integration recommendation (not yet implemented)

For **linear bins**, **`N_dims == 2`**, large `N`:

1. **Kernel:** `tiled128` upper-triangle tile schedule, **2D distance**, **UInt32 counts**, Float32 sums, shared-memory block histogram + global flush.
2. **Do not use:** global `(N,N)` Float32 atomics; global grid-stride for large `N`; Float32 counts.
3. **Sum validation:** compare to **f64 CPU serial** (or device f64 if added), `atol ≈ 500`, not f32 serial gold.
4. **Log / general bins:** still on old ext path until ported with same scheduling ideas + existing `_GPUBinLayout` digitize kernels.

Expected kernel: **~0.0045 s** at `N=20k` vs **~0.096 s** today (~**21×**).

---

## 10. File map (prototype codebase)

| File | Purpose |
|------|---------|
| `GPUPrototypeKernels.jl` | All prototype `@kernel`s, `_pair_from_linear`, tiled macro `@proto_tiled_kernel`, `prototype_variants()` |
| `benchmark_prototypes.jl` | Single-N table, f64 ref validation gate, vs production |
| `benchmark_helpers.jl` | `gold_parity`, timing baseline names |
| `prove_f32_accumulation.jl` | f64 reference proof + f32 order diagnostics |
| `diagnose_sums.jl` / `diagnose_counts.jl` | Focused CPU/GPU reconciliation |
| `ext/StructureFunctionsGPUExt.jl` | **Production** (still slow path) |

---

## 11. Open issues

1. **Per-bin off-by-3** on tiled/blockshared u32 — audit tile partial boundaries; add CPU tiled gold.
2. **Integrate tiled128 2d u32** into `StructureFunctionsGPUExt.jl` for linear + 2D.
3. **Device-resident buffers** in production API (prototype `private_*_device_resident` shows staging win).
4. **Log-spaced bins** on GPU with tile schedule (digitize already in ext; pair schedule not).
5. **Tiled regpriv on CUDA** — needs warp-level or shared-memory privatization without 128KB smem or `MVector` spill; research item.

---

## 12. References

1. Shimizu et al., *CADISHI: GPU-accelerated direct-sum algorithm for particle–particle interactions*, Comput. Phys. Commun. (2018). [doi:10.1016/j.cpc.2018.10.018](https://doi.org/10.1016/j.cpc.2018.10.018)
2. Pitaksirianan et al., *Fast GPU computation of 2-body statistics* (DAPD 2019). [PDF](https://cse.usf.edu/~tuy/pub/DAPD19.pdf)
3. NVIDIA CUDA C++ Programming Guide — shared memory, atomics, occupancy/register pressure.
4. Internal: `src/BinEdges_timings_and_theory.md` — O(1) linear digitize on CPU.
5. Internal: `gpu/prove_f32_accumulation.jl`, `gpu/benchmark_prototypes.jl` — reproducible parity scripts.

---

*Document version: 2026-06-11. Benchmark numbers from user A100 SLURM session (`include_gpu("benchmark_prototypes.jl")`, `N=20000`, seed 42).*
