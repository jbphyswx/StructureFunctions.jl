# Optimal GPU Kernel Design for StructureFunctions.jl

Authoritative design derived from first principles + literature (June 2026). Supersedes
the ad-hoc per-variant kernels. Goal: blazing-fast CPU+GPU for 1D/2D binning, individual
and single-pass (6 invariants), and batched inputs — with **one source of truth per kernel
family** so we never refactor this again.

## 1. The computation

For every unordered pair `(i,j)`, `i<j`, of `N` points with positions `x` and velocities
`u` in `D` dimensions:

```
dx   = x_j - x_i              # separation
r    = |dx|;  r̂ = dx / r      # distance + unit vector
du   = u_j - u_i              # velocity difference
du_L = du · r̂                 # longitudinal
du_norm2 = du·du ;  du_L2 = du_L²
du_T2    = du_norm2 - du_L2   # transverse² (no second dot product needed)
```

Accumulate into a distance bin `b = digitize(r)`:
- **individual** (1 moment, chosen `sf_type`)
- **single-pass** (6 invariants): `{du_norm2, du_L2, du_T2, du_L·du_norm2, du_L·du_L2, du_L·du_T2}`

Output histograms:
- **1D**: `(NMOM, NB)` — distance bins only. `NB ≤ 128`.
- **2D joint**: `(NMOM, n_dist, n_val)` — distance × value bins.

**Batched** adds a trailing `B` axis:
- **fixed-x**: geometry `x` shared across all `B`; only `u` varies → compute `(r,r̂,bin)` ONCE, reuse across `B`.
- **varying-x**: both `x` and `u` vary → `B` independent problems.

## 2. Why the old design failed

- **~47 hand-written kernels** (`{2d,3d}×{linear,log,general}`, 39 `@eval`'d sp2d variants,
  13 batch merge kernels). Fixes must be copy-pasted N× and drift. This is the root cause.
- **Batch joint2d** ran a naive per-cell `(N,N)` global-atomic kernel inside a host
  `for b in 1:B` loop → the ~17s (vs 2–5s target) regression.
- **No replication / contention mitigation** was ever correctly implemented. A prior
  "staggering" attempt offset the *bin index* (corrupts the histogram) instead of the
  *replica index*.

## 3. Optimal algorithm (literature-grounded)

Workload = **N-body tiling fused with a histogram scatter**. Per-pair compute is heavy
(~20–40 FLOPs), so data-reuse tiling matters and atomics are relatively cheap per pair.

1. **Tile** points into blocks of `p` (=128) staged in shared memory; one workgroup per
   **upper-triangular tile-pair** (off-diagonal: full `p²`; diagonal: `j>i` loop). Optional
   **tile-level distance cull** skips tile-pairs whose minimum separation exceeds the max
   bin edge — cell-list-like speedup when bins are cutoff-limited, free otherwise.
2. **1D histogram → privatize + replicate in shared memory.** Hist is tiny
   (6×128×4B ≈ 3KB), so keep `R` independent replicas to cut atomic contention:
   ```
   replica = warp_id % R                      # rotate the REPLICA, never the bin
   slot    = (moment*NB + bin)*R + replica
   ```
   Replicas are full independent histograms, summed order-independently at flush — no
   un-stagger. Pad the layout so consecutive lanes hit distinct banks. Flush = one global
   atomic per `(moment,bin)`. **No separate merge kernel.**
3. **Batch fixed-x → the strip width `W` is the privatization axis.** Compute geometry once
   per pair; accumulate `W` velocity fields into a `(moment×bin×W)` shared histogram; flush
   once. `W` competes with `R` for the 48KB budget. Eliminates the old global-partial+merge.
4. **2D joint histogram (≈393KB) → direct global atomics.** Too big for shared, but spread
   over ~16K cells → low per-cell contention; on Volta+ this is fastest. (Optional shared
   fast-path when `NMOM·n_dist·n_val` actually fits; value-axis tiling only if profiling
   shows L2 contention.)
5. **Reductions** use warp-shuffle butterfly, never serial `lid==1` loops.

## 4. The kernel set (collapse ~47 → ~3)

All specialized at compile time via `Val{}` type parameters (zero runtime cost, confirmed
for KernelAbstractions). Production pins tuned `R`,`W` to one value each → small
specialization matrix (`2 dims × 3 digitizers × 2 nmom × 2 fixed ≈ 24`).

### `sf_tiled_1d!`
Params: `Val{NDIMS}`, digitizer functor `D` (linear/log/general; `general` carries edges),
`Val{NMOM}` (1 or 6), `Val{R}`, `Val{W}` (1 = non-batch/varying-x), `Val{FIXED_X}`.
Covers: non-batch individual + sp1d, fixed-x batch, varying-x batch. No merge kernel.

### `sf_tiled_2d!`
Params: `Val{NDIMS}`, dist digitizer, value digitizer, `Val{NMOM}`, `Val{W}`,
`Val{FIXED_X}`, `Val{SHARED}`. Direct global atomics into `(NMOM,n_dist,n_val[,B])`.
Covers: non-batch joint2d + sp2d, both batch variants (fixes the 17s path with a single
fused launch — no host `for b` loop, no naive per-cell kernel).

### `reduce_partials!`
One parametric warp-shuffle/tree reduction. Replaces all 15 merge kernels. Used only where
global partials genuinely remain (rare given privatization).

## 5. Memory layouts (locked — hard to change later)

- `x`, `u`: `(NDIMS, N)` non-batch; `(NDIMS, N, B)` batch. **Fixed-x `u` staged WITHOUT** the
  old double-`permutedims` to `(B,N,NDIMS)`; stage so the `W`-strip load is coalesced.
- Outputs: 1D `(NMOM, NB[, B])`; 2D `(NMOM, n_dist, n_val[, B])`. `NMOM=1` collapses cleanly.
- Buffers allocated once; reset with `fill!` (async), not `KA.zeros`, inside any loop.
  Async launches, a single `KA.synchronize` at the end.

## 6. Empirical tuning (Slurm hand-off — never run GPU in chat)

Sweep on representative **non-uniform** turbulence fields and lock constants:
`R`, tile `p∈{64,128,256}`, `W`-vs-`R`, block size / occupancy, warp-aggregation on/off,
2D shared-vs-global threshold. A benchmark harness exposes these via the `Val{}` params; the
user runs it in a Slurm allocation and returns numbers.

## 7. Execution phases (tests stay green; correctness validated on `KA.CPU()`)

- **P0** Cleanup stray/experimental files.
- **P1** Building blocks: digitizers, geometry, moments, privatized+replicated accumulator, reduction.
- **P2** `sf_tiled_1d!` + migrate the 4 1D paths; parity-test vs current on CPU.
- **P3** `sf_tiled_2d!` + migrate the 4 2D paths (fixes 17s); parity-test.
- **P4** Unify reductions; delete the 15 merge kernels.
- **P5** Delete the ~40 dead kernels; collapse dispatch.
- **P6** Slurm benchmark harness; tune; lock constants.
- **P7** CPU/threaded batch verification.

## 8. References

KernelAbstractions.jl docs (localmem/synchronize/atomix/performance examples); Gómez-Luna
et al. 2013 (histogram replication R-factor); NVIDIA Maxwell shared-atomics & warp-aggregated
atomics blogs; GPU Gems 3 Ch.31 (N-body tiling); CADISHI arXiv:1808.01478 (parallel pair
distance histograms, diagonal/off-diagonal split); Jia et al. Volta/Turing microbenchmarks
(atomic latency under contention).

---

# FINAL SETTLED DESIGN (measured on A100, 2026-06-23)

After exhaustive prototyping+measurement (see `gpu/benchmark_results/`), the kernel
design is locked. Metric is **billion atom-pairs/s (bapps)** at saturation (CADISHI
ref ~495 for *pure distance*; our SF does more per pair). All numbers N=20000, A100.

## Structure (the decisive finding)
**N-body broadcast loop**, NOT thread-owns-a-pair. Each thread owns its point `i`
in registers and loops `j` over the staged tile so all lanes read the *same*
`shared[j]` each step (broadcast, no bank conflict, NO per-pair `_pair_from_linear`
sqrt). This was a 1.36–2.0× win for 1D and recovers/beats the old code.
- Tile = block: `TILE` points per tile, `TILE` threads/block, thread `t` owns tile point `t`.
- Upper-triangular tile-pairs. Off-diagonal: thread loops `j=1..nj` (broadcast).
  Diagonal: thread loops `j=lid+1..ni` (loses broadcast, but minority of tiles).
- Histogram: per-block **privatized** (shared) + atomic-merge to global at block end.

## Per-regime parameters (all measured)
| regime | structure | TILE | histogram |
|--------|-----------|------|-----------|
| 1D individual (NMOM=1) | N-body | 256 | shared `(NB)`, static |
| 1D single-pass (NMOM=6) | N-body | 256 | shared `(6·NB)`, static |
| 2D joint (NMOM=1) | N-body | 1024 | shared `(n_dist·n_val)` |
| 2D single-pass (NMOM=6) | N-body | 1024 | **dynamic** shared `(6·n_dist·n_val)` sums+counts |

- **TILE=1024 for 2D** (measured, job 238806): on 50×50 the dynamic hist (117KB)
  pins 1 block/SM, so threads/block IS occupancy. 1024→50% occ = **8.36 bapps**
  (192.8s) vs 512→25%=6.39 (252.6s) vs 256→12.5%=3.68. The larger tile's extra
  serial j-work is more than repaid. Static staging at 1024 = 32KB (4 arrays×2×1024×4B);
  32KB+117KB = 149KB < A100 163KB. For small hist all TILEs hit ~100% occ (256≈512≈1024),
  so a single TILE=1024 is safe across 2D. **On smaller GPUs** (L40 100KB) the 1024
  staging+hist may not fit → device-aware fallback drops TILE then static/global.
- **Counts on-chip** (sums+counts both in shared). counts→global measured 3–20× SLOWER.
- **No replication (R=1)**, **no value-axis tiling**, **no warp-aggregation** — all
  measured net-negative for our scattered-bin regime.
- **Dynamic shared** (CC7.0+ opt-in via `CuDynamicSharedArray` + max-dynamic-smem
  attribute). Device-aware: query `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (A100 163KB,
  L40 100KB, V100 96KB). If `6·n_dist·n_val·8 + staging` exceeds it (huge bins on a
  small GPU), fall back to the static-shared/global path.
- fixed-x: identical structure; `x` from `(D,N)` (broadcast across batch), `u` per-b.
- 3D: identical, D=3.

## Implementation note (backend specialization)
KA `@localmem` is static-only (≤48KB) and KA exposes no warp/dynamic-shared ops, so
the fast path is **CUDA-specialized** (a CUDA kernel via `@cuda` + `CuDynamicSharedArray`),
dispatched on the CUDA backend type. The portable KA kernels remain as the CPU
reference. `GPUBackend{B}` is parametric precisely to allow this.

## Final expected times @ N=20000, B=8064 (extrapolated, single-pass unless noted)
| path | bins | time |
|------|------|------|
| 1D individual | NB=50 | ~10–11 s (152 bapps) |
| 1D single-pass | NB=50 | ~50–60 s |
| 2D joint (1 SF) | 50×50 | ~31 s |
| 2D single-pass | 16×8 | ~120 s |
| 2D single-pass | 20×20 | ~135 s |
| 2D single-pass | 50×50 | **~58 s** (TILE=1024, linear value bins) — ~17× the old 974 s |

**Integrated-path timing (job 238895, public API, N=20000):** 50×50 SP2D fixed-x =
0.457 s @ B=64 = **28 bapps → ~58 s @ B=8064**. This beats the proto's 192.8 s
estimate because the proto used a *general* (binary-search) value plan; the real
workload's `LinearBinEdges` value bins hit the **O(1) FMA digitize**
(`GPUValueLinearShared`), ~3× faster in the 6-per-pair value-digitize hot loop.
Lesson: benchmark with the production value-plan type, not a general fallback.

2D-SP is intrinsically heavy (6 invariants × 2D × 1.6e12 pairs); 50×50 ≈ 252 s is
near its practical floor on this hardware with the on-chip-histogram approach.

---

# IMPLEMENTED + VALIDATED (final state, jobs 238837/238895/239150/239164/239175)

`StructureFunctionsCUDAExt = ["KernelAbstractions","CUDA"]` holds the CUDA fast
kernels (`ext/cuda/kernels_{1d,2d}.jl`); it overrides package stubs
`gpu_fast_launch_{1d,2d}_batch!` (default `false` → KA fallback) on
`::CUDA.CUDABackend`. GPUExt's `_sf_launch_{1d,2d}_batch!` call the hook first.
Shared device-callable building blocks (`_sf_moments`, `_sf_dot`,
`_gpu_digitize_value_plan`, digitizers, value plans) reused from GPUExt via
`const GE = Base.get_extension(...)`. Batch + slices dispatch all funnel through
the unified launchers; non-batch (B=1, sub-10 ms) stays on the existing tiled128
path. Full CPU suite 4374/4374 green.

## Final per-regime routing (measured-optimal, histograms verified equal)
| regime | kernel | bapps | @ B/T=8064 |
|--------|--------|-------|-----------|
| 1D individual **fixed-x** | OLD global warp-replica + W-strip | 147 | 11 s |
| 1D individual varying-x | **N-body** broadcast | 115 | 14 s (1.9× old 61) |
| SP1D (fixed + varying) | **N-body** | 43 | 37 s (1.4× old 30; beats 6×separate=84 s) |
| joint 2D (fixed + varying) | **N-body** + shared hist | — | — |
| **SP2D 50×50** (fixed + varying) | **N-body** + dyn-shared, TILE=1024 | 28 | **58 s (~17× old 974 s)** |

Decisive finding: **the optimum is regime-dependent**. 1D-individual-fixed-x has a
tiny histogram and is contention-bound → the old global-warp-replica design (R=8
global replicas, geometry amortized over a W-strip) beats every shared-histogram
N-body variant (incl. an N-body + W-strip kernel, which measured *slower* at 108
bapps — geometry amortization doesn't pay when occupancy/contention dominate). All
other regimes are geometry- or atomic-bound → N-body broadcast (+ dynamic-shared
for large 2D) wins, by up to 17×. "One kernel for everything" is NOT optimal; the
dispatch routes each regime to its measured winner.

Parity: exact integer counts across all regimes on GPU vs CPU; sums within FP32
reduction-order tolerance (≤ ~1e-3 relative on near-cancelling single-pass cells;
≤ ~1e-5 elsewhere). GPU↔CPU count parity is exact except at value-bin edges where
FMA rounding can move a handful of pairs — use `atol`, not exact equality, for
GPU-vs-CPU sum comparisons.

---

# Lessons & RULED-OUT approaches (measured — do NOT re-try)

Every item below cost at least one A100 run. They are settled; re-deriving them
wastes GPU time. Raw evidence is in `gpu/benchmark_results/` (job IDs cited).

## Algorithmic levers that were tested and LOST
| approach | result | why it loses |
|----------|--------|--------------|
| Shared-histogram **replication** R>1 (1D) | R1=33, R2=34, R4=23, R8=12, R16=PTX overflow (NMOM6,N8000,B16) | extra shared replicas cut occupancy; A100 shared atomics are fast enough that contention-spreading isn't worth it |
| Fixed-x **W-strip** on the *shared-hist* 1D kernel (NMOM=6) | W1=34, W2=31, W4=22, W8=12 | same — striping blows occupancy for the 6× histogram |
| **counts → global** (sums on-chip, counts global), 2D-SP | 0.6–2.0 vs 12.6 bapps both-on-chip (job 238789) | global count-atomics dominate; occupancy gain from halving shared not worth it. **Keep sums+counts BOTH on-chip.** |
| **Value-axis tiling** (split value bins into slabs for occupancy), 2D-SP | slower everywhere (jobs 238805/238806) | nslabs× pair re-traversal > occupancy gain |
| **N-body + W-strip geometry amortization**, 1D-individual-fixed | 108 vs 115 bapps plain N-body (job 239154) | geometry amortization doesn't pay when the regime is contention/occupancy-bound, not geometry-bound |
| **Warp-aggregated atomics** (`match.any.sync`) | not pursued — research-confirmed marginal | only pays when warp lanes share few bins; our bins are scattered → degenerates to ~32 atomics. `match_any_sync` also absent from the CUDACore stack (reachable only via raw llvmcall). `redux.sync.add` is integer-only. |
| Unified "one kernel for all regimes" | 1D-individual-fixed regressed ~28% | **the optimum is regime-dependent** — see routing table above. Dispatch per regime. |
| Old varying-x 2D (`sf_tiled_2d_varying!` direct global atomics) | 7× slower than shared-hist (job 238782) | 2D needs a privatized (shared/dynamic-shared) histogram, not direct global atomics |

## Things that WON (and the decisive evidence)
- **N-body broadcast** (thread owns point i, loops j over staged tile → all lanes read same `shared[j]`): 1.36–2.0× over thread-owns-a-pair for geometry-bound 1D (job 238798). The ~7× gap to CADISHI was *structure* (scattered reads + per-pair sqrt-decode), not atomics.
- **Dynamic shared memory** (`CuDynamicSharedArray` + `FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`): unlocks the full single-pass 2D histogram on-chip (A100 163 KB opt-in vs 48 KB static). This is what makes 50×50 SP2D fit.
- **TILE=1024 for 2D**: 50% occupancy beats 25% (TILE=512) on the histogram-pinned 50×50 case — 8.36 vs 6.39 bapps (job 238806).
- **O(1) FMA value digitize** (`GPUValueLinearShared` from `LinearBinEdges`) vs general binary-search (`GPUValueVectorCols`): ~3× in the 6-per-pair hot loop → 58 s vs the proto's 193 s estimate. **Always benchmark with the production value-plan type, not a general fallback.**
- **Single fused launch** for batch/slices (no host `for b`/`for slice` loop): the original ~17 s batch-joint2d regression was a naive per-cell `(N,N)` kernel inside a host loop.
- **Stagger the REPLICA index, never the bin index** (if replication is ever used): offsetting the bin corrupts the histogram — the prior botched attempt.

## Compile pitfalls (KA.CPU() catches NONE — only `@cuda` compile does)
- `threadIdx().x`/`blockIdx().x`/`blockDim().x` return **Int32** → convert to `Int` at kernel top, or device helpers annotated `::Int` throw `jl_f_throw_methoderror`. (A ternary `lid+1 : 1` promotes to Int64, so the *second* use can compile while the first raw-Int32 use throws.)
- `zero(eltype(localmem))` / `zero(::DataType)` doesn't constant-fold → methoderror. Use `zero(FT)` with `FT` a concrete type param.
- Looped reads/writes/atomics on a shared array passed as a **function argument** fail to compile on the CUDACore stack — write them **inline** in the kernel body (single-element `@inline` loads are fine).
- **Dynamic-shared opt-in must be set when STATIC+DYNAMIC shared > 48 KB**, not only when the dynamic part alone exceeds it (job 238836 bug: NMOM=1 50×50 had 20 KB dynamic + 32 KB static staging = 52 KB > 48 KB default cap → launch failed until the attribute was set unconditionally).
- Non-bitstype kernel args (e.g. an untyped `edges_dev::Any` field) → KernelError; type the field + add an `Adapt.adapt_structure` rule.
- Workflow that catches everything but InvalidIRError for free, on the login node (no GPU): `Meta.parseall` + `using StructureFunctions, KernelAbstractions, CUDA` (extension load/precompile). Only `@cuda` compile (sbatch) catches InvalidIRError.

## Reproducing the validation (each is a self-contained `gpu/` script + `run_*.sh`)
- `gpu/test_cuda_2d_parity.jl` — 2D kernel parity (CUDA vs KA.CPU), 12 combos.
- `gpu/test_cuda_1d_parity.jl` — 1D kernel parity + timing + joint2d-varying FP-boundary check.
- `gpu/bench_1d_old_vs_nbody.jl` — the head-to-head that decided 1D per-regime routing.
- `gpu/test_e2e_2d_cuda.jl`, `gpu/test_slices_e2e.jl` — public-API parity vs serial + timing.
