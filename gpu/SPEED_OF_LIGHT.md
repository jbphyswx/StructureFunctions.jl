# Speed-of-light analysis and exploration plan (GPU pair kernels)

**Status: working document. Nothing here is gospel — including this sentence.**

Every claim is tagged so a later reader knows what it rests on:

- **[M]** measured this session, with the method stated so it can be re-run and falsified.
- **[D]** derived from first principles / algebra, with the derivation shown.
- **[I]** inherited claim from `OPTIMAL_KERNEL_DESIGN.md` or code comments, **not re-verified here**.
- **[H]** hypothesis, untested, with a stated kill criterion.

Companion to `OPTIMAL_KERNEL_DESIGN.md`, which records the June 2026 design. That document's
measurements were taken on a different code state and several of its conclusions are premised on
assumptions that later work can invalidate (see §5). Treat it as background, not as fact.

Hardware for all **[M]** numbers: NVIDIA A100-SXM4-80GB, 108 SMs, 1.41 GHz, 164 KB shared/SM,
65536 registers/SM, 163 KB shared/block opt-in. Julia 1.12.7, CUDA.jl as pinned in `gpu/`.

---

## 1. What the workload actually is

Per unordered pair `(i,j)`, `D=2`, single-pass (6 invariants):

| stage | cost |
|---|---|
| `dx = x_j - x_i`, `r² = dx·dx`, `r = sqrt(r²)` | ~5 FLOP + 1 sqrt |
| `bin = digitize(r)` (FMA route) | ~2 FLOP |
| `du = u_j - u_i`, `du_L = du·dx/r`, `|du|²` | ~8 FLOP |
| 6 moments | ~4 FLOP |
| **accumulate** | **6 shared sum-atomics + 1 shared count-atomic** |

So ~20 FLOP and **7 shared-memory atomics** per pair. The FLOPs are free; the atomics are not.

## 2. Validated cost model: atomics ≈ 56%, fixed geometry ≈ 44% **[M]**

Measured by holding the kernel family fixed (N-body, **varying-x** so both NMOM route to the same
kernel) and varying only `NMOM`. Float32, `N=20000`, `LinearBinEdges(range(0, 0.3; length=33))`,
`CUDA.@elapsed` over 3 iterations after warmup.

| B | NMOM=1 (2 atomics/pair) | NMOM=6 (7 atomics/pair) | t6/t1 |
|---|---|---|---|
| 1 | 106.8 bapps | 64.2 | 1.66 |
| 8 | 119.7 | 71.3 | 1.68 |
| 64 | **149.3** | **89.1** | **1.68** |

Fit `t_per_pair = a + b·n_atomics` at B=64: `a = 0.00489`, `b = 0.000905` (bapps⁻¹).

- **atomic-attributable share at NMOM=6: ~56%**
- **fixed share (geometry, sqrt, shared loads, loop): ~44%**

**[D] Consequences.** Removing atomics *entirely* would give at most `1/0.436 ≈ 2.3×`. Any
atomic-reduction lever (§3, §4) is therefore bounded by ~2×, not by an order of magnitude. The other
~44% is only reachable by attacking geometry cost and occupancy (§8).

### Why the naive roofline framing is wrong — a falsified analysis, kept as a warning **[M]**

An earlier version of this section computed: A100 shared-atomic ceiling
`108 SM × 1.41 GHz × 32 banks = 4.9e12 ops/s`; our 64 bapps × 7 atomics = 4.5e11 = **9%** of it;
CADISHI's ~495 bapps × 1 atomic ≈ 10%. It concluded there was ~10× of atomic headroom.

**That conclusion does not follow, and the measurement above refutes it.** Being at 9% of the
*atomic* ceiling is irrelevant when atomics are only ~56% of runtime — the bound is ~2×.

Two experimental errors produced it, both worth avoiding again:

1. **Comparing across kernel families.** The first attempt compared NMOM=1 against NMOM=6 using
   `x=(D,N)`, `u=(D,N,1)`. That is the **fixed-x** regime, which routes to a completely different
   *global*-atomic warp-replica kernel. It measured 37.3 bapps for NMOM=1 vs 64.5 for NMOM=6 — i.e.
   *more* atomics running *faster* — which is a routing artifact, not a hardware property. Always
   confirm both arms hit the same kernel before attributing a difference to one variable.
2. **Comparing an unsaturated point to a saturated reference.** `B=1` does not saturate the device;
   the bapps figures in `OPTIMAL_KERNEL_DESIGN.md` are at `B/T=8064`. Comparing our B=1 number to
   CADISHI's saturated ~495 bapps is not a like-for-like comparison.

## 3. Lever A — two of the six moments are exactly redundant **[D]**

The six accumulated invariants are not independent:

```
S2   = |δu|²                 L2 = δu_L²
T2   = |δu|² − δu_L²                        =  S2 − L2
S3   = δu_L·|δu|²            L3 = δu_L·δu_L²
L1T2 = δu_L·T2 = δu_L·(|δu|² − δu_L²)       =  S3 − L3
```

Histogram accumulation is linear, so **per bin** `Σ T2 = Σ S2 − Σ L2` and `Σ L1T2 = Σ S3 − Σ L3`
exactly. `T2` and `L1T2` never need a per-pair atomic; they are one subtraction each at flush,
over `NB` bins rather than over `N²/2` pairs.

**Atomics per pair: 7 → 5.** Using the validated cost model of §2 (`a = 0.00489`, `b = 0.000905`):
`t = a + 5b = 0.00941` → **106.2 bapps predicted vs 89.1 baseline = 1.19×**, at B=64 Float32.
(An earlier estimate of ~1.4× assumed atomics dominated; §2 shows they are ~56%.)

### IMPLEMENTED and measured — CUDA 1D N-body kernel **[M]**

`_sf_accum_moments` / `_sf_flush_moment` in `ext/gpu/sf_core.jl`; applied in
`ext/cuda/kernels_1d.jl`.

| | before | after | speedup |
|---|---|---|---|
| Float32, B=64 | 89.1 bapps | **105.1 bapps** | **1.18×** |
| Float32, B=1 | 3.12 ms | 2.66 ms | 1.17× |
| Float64, B=1 | 5.35 ms | 4.50 ms | 1.19× |

**Predicted 1.19×, measured 1.18× — the cost model of §2 is accurate to ~1%**, which is the more
valuable result: it can now be used to rank the remaining levers before writing code.

Precision, against a CPU serial reference, with the KA path (which still accumulates all six
directly) as the control:

| moment | direct-6 | reconstructed-4 |
|---|---|---|
| T2 Float32 | 3.89e-5 | 3.82e-5 |
| L1T2 Float32 | 5.97e-5 | 7.82e-5 |
| T2 Float64 | 2.20e-14 | 2.09e-14 |
| L1T2 Float64 | 1.81e-14 | 2.08e-14 |

Worst case (L1T2/F32, 7.8e-5) is comparable to the directly-accumulated L3 (5.5e-5) and well inside
the documented ~1e-3 near-cancelling tolerance. Kill criterion not triggered. CUDA suite 29/29.

### Scope limit — does NOT apply to SP2D **[D]**

In single-pass **2D**, each moment is binned by **its own value** on the value axis
(`vbin = digitize(vals[t])`, `ext/gpu/kernels_2d_direct.jl`). The pairs contributing to cell
`(t=3, dbin, vbin)` are selected by `T2`'s value, not by `S2`'s or `L2`'s, so
`Σ_cell T2 ≠ Σ_cell S2 − Σ_cell L2`. **The reconstruction is only valid where all moments share
one bin, i.e. sp1d.** Do not propagate it into any 2D value-axis kernel.

Still to propagate (same algebra, same validity condition): the KA sp1d kernels
(`ext/gpu/kernels_1d_single_pass.jl`, `ext/gpu/sf_tiled.jl`, the global-atomic fallbacks), which
serve the **non-batch** path, and the CPU `_sp1d_pairs!` (6 stores → 4, plus the redundant 6×
identical count writes → 1).

Caveats to measure, not assume:
- **Precision.** `Σ(a−b)` and `Σa − Σb` differ in rounding. `T2` is a difference of same-sign
  quantities and can be near-cancelling; `OPTIMAL_KERNEL_DESIGN.md` already reports ~1e-3 relative
  FP32 error on near-cancelling single-pass cells **[I]**. This change could worsen that. **Kill
  criterion:** if FP32 relative error on `T2`/`L1T2` degrades beyond the existing GPU-vs-CPU
  tolerance on a turbulence-like field, keep 6 accumulators in FP32 and apply the reduction only
  for FP64, or accumulate `T2` directly and derive `L1T2` only.
- Applies identically to CPU, where the 6 moment stores are plain writes but still 6× the traffic.
- **[M]** The CUDA N-body kernel already issues only **one** count atomic (`scnt[bin] += 1`,
  `ext/cuda/kernels_1d.jl`), so counts are not a redundancy there. The CPU `_sp1d_pairs!` loop
  writes the same count 6× (`for t in 1:SINGLE_PASS_N; counts[t,bin] += 1`) — free to fix.

## 4. Lever B — spatial sorting changes the premise under which warp aggregation was rejected **[H]**

`OPTIMAL_KERNEL_DESIGN.md` rejects warp-aggregated atomics: *"only pays when warp lanes share few
bins; our bins are scattered → degenerates to ~32 atomics"* **[I]**.

That rejection is not a statement about the algorithm — it is a statement about **the input
ordering**, which is an artifact we control. If points are spatially sorted (Morton/Hilbert order,
or cell-list order), then in the N-body inner loop thread `i` walks `j` through spatially adjacent
points, so `r = |x_i − x_j|` varies smoothly and consecutive `j` land in the *same or adjacent* bin.

Three things unlock from the same preprocessing pass, and none of them work without it:

1. **Run-length accumulation in registers.** A thread accumulates while the bin is unchanged and
   issues one atomic per run instead of one per pair.
2. **Warp aggregation becomes viable** — the rejected optimization, under its stated precondition
   now satisfied rather than violated.
3. **Tile-level distance culling** (already noted as "optional" in `OPTIMAL_KERNEL_DESIGN.md` §3.1
   but never implemented) becomes effective, because a spatially sorted tile has a tight AABB.

**[M]** Independent measurement supporting (3): at `r_max/L = 0.05`, only **0.75%** of pairs fall
in range (exact count over all 2.0e8 pairs, N=20000 in a unit square); at 0.10, 2.9%. Nothing in the
current code prunes — `for j in (i+1):n_points` visits every pair and discards after computing the
distance (`src/Calculations/serial_single_pass.jl:311-315`).

**Kill criterion:** if, on a *non-uniform* (turbulence-like) field with realistic bin edges, the
mean run length at the chosen tile size is < 2, drop lever B entirely — the sort cost and the
register pressure will not be repaid.

Note this composes with, and largely subsumes, the "Stage 7 cell-list" item: both need one
spatial sort, and sorting is safe because a structure function is a sum over *unordered* pairs, so
any permutation of points leaves the histogram identical up to FP summation order.

## 5. Inherited claims that are premise-dependent — re-test before trusting **[I]**

These are recorded as settled in `OPTIMAL_KERNEL_DESIGN.md`. Each is conditional on something that
later work can change; none were re-measured this session.

| inherited claim | the premise it rests on | why it may not hold |
|---|---|---|
| replication R>1 loses (R1=33, R2=34, R4=23, R8=12) | shared memory is scarce, so replicas cost occupancy | **[M]** sp1d Float32 is *register*-bound (5 blocks/SM by registers vs 21 by shared) — shared is slack there, so the stated cost mechanism does not apply in that regime |
| warp aggregation is marginal | "our bins are scattered" | bin scatter is a property of input ordering (§4), not of the algorithm |
| tile-level distance cull "optional… free otherwise" | never implemented or measured | **[M]** 0.75% in-range pairs at `r_max/L=0.05` is not "free otherwise" territory |
| non-batch stays on tiled128 because it is "sub-10 ms" | that sub-10 ms does not matter | **[M]** it is 1.24–1.58× slower than the N-body path in Float32 (§6) |
| constant memory | **never tried — absent from the ruled-out table** | see §7 |

## 5b. BUG FOUND AND FIXED: Float64 fixed-x batch never compiled **[M]**

`_batch_fixed_x_usmem_priv!` stages `512 + 512·W` elements of `FT` in **static** `@localmem` with a
hardcoded `BATCH_USMEM_STRIP_W = 16`. That is an unstated Float32 assumption:

| FT | staging | 48 KB static cap |
|---|---|---|
| Float32 | 8704 × 4 = 34,816 B | fits |
| Float64 | 8704 × 8 = **69,632 B** (`0x11000`) | **exceeds `0xc000`** |

So `calculate_structure_function(sf, x::CuArray{Float64,2}, u::CuArray{Float64,3}, bins)` — fixed-x
batch, Float64 — died at launch with `ptxas error: uses too much shared data (0x11000 bytes, 0xc000
max)`. The path had therefore **never executed**, so its numerics had never been validated either.

Fixed by making the strip width a budget computation rather than a constant —
`_batch_usmem_strip_w(FT) = clamp(GPU_SMEM_STATIC_MAX ÷ (512·sizeof(FT)) − 1)`, rounded down to a
power of two: 16 for Float32 (unchanged), 8 for Float64. Verified after the fix against a CPU serial
reference: **counts exact**, sums 1.5e-14 (F64) / 2.0e-5 (F32) relative, at B=1 and B=5.

**Lesson for the next reader:** this is the second bug of exactly this shape (the first was static
staging checked against the *opt-in* limit instead of the *static* cap). Any `@localmem` sized from
a compile-time constant is a Float64 hazard — size it from `sizeof(FT)` against
`GPU_SMEM_STATIC_MAX`, and test Float64, which is where these surface.

## 6. Measured: the non-batch entry points run the slower structure **[M]**

Same workload, Float32, N=20000. Non-batch calls run the KA thread-owns-a-pair kernel; passing
`reshape(u, D, N, 1)` routes to the batch launcher, which reaches the CUDA N-body kernel.

Full per-regime A/B, N=20000, after Lever A and the §5b fix. Ratio > 1 means routing non-batch
through the batch launcher wins:

| regime | Float32 | Float64 | route non-batch to batch dispatcher? |
|---|---|---|---|
| 1D individual (L2SF) | **0.64×** | **0.76×** | **NO — non-batch wins in both** |
| 1D single-pass | 1.90× | 1.32× | yes |
| joint 2D | 1.48× | 1.18× | yes |
| single-pass 2D | 1.24× | **2.58×** | yes |

Verified equivalent, not merely faster: worst relative difference across all six invariants is
**7.7e-14 (Float64)** and **3.3e-4 (Float32)** — a different summation order over ~4.8e7 in-range
pairs, nothing more.

### Why 1D-individual loses on the batch route — the mechanism, not just the ratio **[M]**

From `CUDA.@profile`, Float32, N=20000:

| arm | kernels launched | main kernel | regs | smem |
|---|---|---|---|---|
| non-batch (3.45 ms) | **1** — `_sf_kernel_tiled128_2d_linear_u32` | 3.28 ms | 38 | 5,120 B |
| batch B=1 (5.42 ms) | **4** — permutedims + main + 2 merges | 3.84 ms | 50 | 34,816 B |
| sp1d batch B=1 (2.63 ms) | **1** — `_cuda_sf_1d_kernel` (N-body) | 2.53 ms | 38 | 11,776 B |

Three causes, additive:

1. **It never reaches the N-body kernel.** sp1d batch lands on `_cuda_sf_1d_kernel` (grid 3160 →
   TILE=256, N-body); individual fixed-x batch lands on the KA `_batch_fixed_x_usmem_priv`
   (grid 12403 → TILE=128). The 1.90× and the 0.64× are not the same comparison.
2. **Four kernels instead of one** — a `permutedims`, the main kernel, and two merge kernels
   (0.40 ms) plus per-launch overhead.
3. **The main kernel is slower at B=1** (3.84 vs 3.28 ms) with 50 vs 38 registers and **6.8× the
   shared memory**, because it stages a width-16 velocity strip to serve a batch of one.

**[D] The unifying reason:** the fixed-x batch path exists to amortize per-tile geometry across a
wide batch strip. At `B=1` there is nothing to amortize, so strip staging, block-private partials
and the merge kernels are pure overhead. This *reconciles* rather than contradicts
`OPTIMAL_KERNEL_DESIGN.md`, whose 147-bapps figure for this kernel is at `B=8064` where the
amortization does pay — so the correct statement is that the routing decision is **B-dependent**,
not purely regime-dependent, which is sharper than what the older document says.

**[H] Latent optimization, not taken:** when `bw == 1` the strip machinery and both merge kernels
are provably unnecessary and the batch path could shed them. Not rerouting achieves the same result
more cheaply, so this is only worth doing if the `B=1`-through-batch path is ever needed for another
reason.

Note the sp1d Float32 ratio rose 1.58× → 1.90× after Lever A, because Lever A speeds up the batch
arm. `1.58 × 1.18 = 1.86` vs 1.90 measured — the two levers compose as the model predicts.

## 6b. IMPLEMENTED — sp1d non-batch reroute **[M]**

`_sp1d_try_fast_batch!` in `StructureFunctionsKernelAbstractionsExt.jl`, called as the first
statement of all three `_launch_single_pass_kernel!` route methods (linear / log / general). It
offers the launch to `SFC.gpu_fast_launch_1d_batch!` as `B=1`; a `false` return means the hook
declined and the caller continues to the tiled128 path unchanged. This is the same
hook-or-fall-through pattern `_sf_launch_1d_batch!` already uses, so non-CUDA backends — where
there is no measurement — are untouched by construction.

| FT | route | before | after |
|---|---|---|---|
| Float32 | linear | 5.00 ms | **2.63 ms (1.90×)** |
| Float32 | log | — | 3.42 ms |
| Float32 | general | — | 4.34 ms |
| Float64 | linear | 5.94 ms | **4.64 ms (1.28×)** |

**1D-individual is deliberately NOT rerouted** (0.64×/0.76×, mechanism in §6).

### joint2d and sp2d reroutes — also implemented **[M]**

`_joint2d_try_fast_batch!` (3 route methods in `StructureFunctionsKernelAbstractionsExt.jl`) and
`_sp2d_try_fast_batch!` (`ext/gpu/launch.jl`), same hook-or-fall-through pattern. `sp2d` respects
`force_global_atomic` — an explicit request for the global path is not overridden. joint2d declines
when `val_plan === nothing` (no plan to hand the batch kernel).

| regime | FT | before | after | speedup |
|---|---|---|---|---|
| joint 2D | Float32 | 4.39 ms | 2.95 ms | 1.49× |
| joint 2D | Float64 | 5.24 ms | 4.51 ms | 1.16× |
| single-pass 2D | Float32 | 10.30 ms | 8.28 ms | 1.24× |
| **single-pass 2D** | **Float64** | **27.98 ms** | **10.90 ms** | **2.57×** |

joint2d's `(n_dist, n_val)` output reshapes to `(1, n_dist, n_val, 1)` free of charge — column-major
layout is byte-identical.

### Counts are REDISTRIBUTED, not lost — how to check this properly **[M]**

The sp2d reroute changes per-cell counts for exactly `T2` and `L1T2`, the two moments built on the
cancelling `|δu|² − δu_L²`, so their computed value can land on the other side of a **value**-bin
edge. Measured (Float32, N=800, 26495–48300 counted pairs per moment):

| moment | Σ\|Δcount\| KA.CPU (not rerouted) | Σ\|Δcount\| CUDA (rerouted) | total conserved? |
|---|---|---|---|
| S2, L2, S3, L3 | 0 | 0 | identical |
| T2 | **4** (pre-existing) | 14 | **yes** |
| L1T2 | 0 | 12 | **yes** |

`T2` already differed before any reroute, so that part is pre-existing; `L1T2` is new. **The
decisive check is the total, not the per-cell diff** — a per-cell diff cannot distinguish "pair
moved to the adjacent bin" from "pair lost". Totals match exactly for all six moments, so ~7 pairs
per moment (0.015%) shifted between adjacent value bins and none were lost or double-counted. Sums
move by 5.0e-6 (F32) / 7.0e-15 (F64) relative. Verified: CUDA 29/29, KA/CPU 951 assertions.

Any future change to 2D value binning should be judged the same way: **compare totals first**, then
per-cell.

### Accuracy: judge by a control, not by the raw relative error **[M]**

A naive "worst relative error vs CPU" reads 3.0e-3 for Float32 and looks like a regression. It is a
denominator artifact. Each moment is an array over distance bins; the odd-order moments are sums of
**signed** terms, so an individual bin can nearly cancel to zero while the moment as a whole is
O(0.1). Dividing by such a bin inflates a negligible absolute difference.

Measured (Float32, N=3000, seed 3, 16 bins). "typical magnitude" = `max|value|` over that moment's
own bins:

| moment | worst rel | value at that bin | typical magnitude | abs Δ |
|---|---|---|---|---|
| S2 (sum of squares, no cancellation) | 6.0e-6 | 4.08 | 4.11 | 2.4e-5 |
| L2 | 1.6e-5 | 2.02 | 2.05 | 3.3e-5 |
| T2 (reconstructed) | 2.0e-5 | 2.03 | 2.07 | 4.1e-5 |
| L3 (sum of cubes, signed) | 3.1e-3 | **1.18e-4** | **1.61e-1** | 3.6e-7 |

L3's worst bin is 1371× below that moment's typical magnitude — millions of signed cubes of order
0.1 cancelling down to 1e-4. Every bin of S2 is ≈4.08 by contrast, because squares cannot cancel.

**Do not conclude from this that "divide by the typical magnitude instead" is the right metric** —
that reasoning can equally hide a real defect, and a near-cancelling bin genuinely carries only a
couple of trustworthy Float32 digits on *any* hardware. What actually settles whether a change hurt
accuracy is a **control on identical data**: accumulate all six directly vs accumulate four and
reconstruct two, both against the same CPU reference. That gave **6.0e-5 vs 7.8e-5** worst case —
the change moved the error ~30%, not 50×. Use that comparison, not the raw ratio, to judge any
future change here. Verified: CUDA 29/29, KA/CPU 842 assertions.

## 6c. CPU: Lever A applied, and an anomaly it exposed **[M]**

Lever A also applies to the CPU sp1d kernels, where the stores are plain writes rather than atomics.
Both `_sp1d_pairs!` (scalar) and `_pf_sp_simd_pairs!` (the Euclidean D∈{2,3} SIMD scatter — the
common path) now store 4 sums + 1 count and call `_sp1d_derive_rows!` at the **end of the kernel**.

Placing the derive inside the kernel rather than at the callers is what makes this safe: there are
20+ accumulator assembly points across serial/threaded/MPI/distributed/batch-leading, several of
them sp2d (where Lever A is invalid, §3) or batch-leading with a different axis order. The derive
uses `=`, never `+=`, so it is idempotent and correct however many times a kernel runs over a
buffer, and correct under partial/threaded reduction because it is linear.

**Measured: sp1d 6.00 → 4.57 ns/pair = 1.31×** (N=4000, Float64, 32 bins, serial), better than the
1.22× a scalar micro-benchmark predicted, because the SIMD path gains more.

### Anomaly: sf1d (1 moment) is SLOWER than sp1d (6 moments) **[M]**

| | ns/pair (min, N=4000) |
|---|---|
| sf1d — one invariant | **5.03** |
| sp1d — six invariants | **4.64** |

Reproducible at N=2000 and N=4000. Computing and storing six invariants is ~10% *cheaper* than one,
which means sf1d carries structural overhead sp1d does not. The difference is that
`_pf_simd_pairs!` (sf1d, `serial.jl`) keeps a **per-`i` local histogram**: `fill!` two `nb+2`
buffers, scatter branchlessly via `clamp` into guard cells, then merge `nb` bins into the output —
every `i`. `_pf_sp_simd_pairs!` (sp1d) instead branches (`if 1 <= bin <= nb`) and scatters straight
into the output.

Isolated micro-benchmark of the two scatter strategies, identical compute (N=4000, Float64):

| nb | per-`i` local hist | direct branchy scatter | ratio |
|---|---|---|---|
| 8 | 10.96 ns | 9.78 ns | 1.121 |
| 32 | 10.50 ns | 9.80 ns | 1.071 |
| 128 | 10.28 ns | 9.96 ns | 1.032 |
| 256 | 10.25 ns | 9.79 ns | 1.047 |

**Direct scatter wins at every bin count tested — no crossover**, so this is not regime-dependent.

**[H] Mechanism NOT established — do not repeat this claim without evidence.** The obvious
explanation (the `2(nb+2) + nb` fill/merge ops per `i`) predicts a penalty that *grows* with `nb`.
The measured gap is roughly constant at ~0.6 ns/pair from nb=8 to nb=256, which contradicts it. The
extra buffer round-trip is a better candidate, but that is a hypothesis, not a finding. The
optimization is justified by the measurement regardless; the *reason* is open.

### APPLIED — `_pf_simd_pairs!` converted to direct scatter **[M]**

`clamp(bin, 0, nb+1) + 1` into guard cells 1 and `nb+2` (dropped at merge) is exactly equivalent to
`if 1 <= bin <= nb`, so the scatter now writes straight into `output`/`counts` and the `hloc`/`cloc`
buffers, their allocations, and the per-`i` merge loop are gone (2 call sites: `serial.jl`,
`StructureFunctionsOhMyThreadsExt.jl`).

| | before | after | |
|---|---|---|---|
| sf1d (1 moment) | 5.03 ns/pair | **4.35** | **1.16×** |
| sp1d (6 moments) | 4.64 | 4.75 | — |

The anomaly is resolved: the one-moment kernel is now correctly *cheaper* than the six-moment one,
and sf1d also beats the 4.47 ns/pair session baseline. Verified: 4414 assertions across 16 files,
plus JET 48/48, Aqua 10/10, `test_parallel_equivalence` 14/14 — zero failures.

**Ordering lesson worth keeping:** this was found only by benchmarking two kernels *against each
other* and noticing the result was physically impossible (more work, less time). Absolute timings
against a baseline would not have surfaced it — the baseline for sf1d (4.47) looked fine.

## 7. Constant memory: what CADISHI does, and when it could pay here **[D]**

CADISHI (`bio-phys/cadishi`, `cadishi/kernel/c_cudh_functions.cu`) holds **coordinates in the 64 KB
`__constant__` bank** and gives shared memory entirely to histogram bins:

```c
const int histo_advanced_cmem_bytes = 64000;
__constant__ char histo_advanced_coords_cmem[histo_advanced_cmem_bytes];
extern __shared__ uint32_t smem_bins[];
cudaMemcpyToSymbol(histo_advanced_coords_cmem, &coord_d[iOffset], cmem_tile_n_elem*sizeof(TUPLE3_T));
```

Tile = `64000/sizeof(TUPLE3_T)` ≈ 5333 points (float3) vs our 128-point shared tile. It also blocks
the *bin range* across `blockIdx.y` (`bin_lo = smem_nbins * blockIdx.y`) instead of falling back to
global atomics when bins exceed shared capacity.

**[D] Where this can and cannot pay.** Shared memory already broadcasts at full speed when every
lane reads the same address — which is exactly the N-body access pattern — so the win is *not* read
latency. The win is that coordinates stop consuming shared memory. Therefore it helps **only where
shared memory is the binding constraint**:

- **1D single-pass: no.** **[M]** register-bound at 5 blocks/SM; shared would allow 21. The ~4.5 KB
  reclaimed is already slack.
- **SP2D 50×50: plausibly yes.** **[I]** the 117 KB dynamic histogram "pins 1 block/SM, so
  threads/block IS occupancy". Moving 32 KB of staging out of shared is the difference between 1 and
  2 blocks/SM, in the one regime the design doc calls "near its practical floor".

### DEAD LEVER — measured, do not retry **[M]**

**The precise CUDA.jl blocker.** The primitives exist: `AS.Constant = 4` (`pointer.jl`), `CuGlobal`
+ `cuModuleGetGlobal_v2`, and a working example in `random.jl`. But `emit_constant_array` bakes the
data into the LLVM IR as a compile-time `ConstantArray` initializer — fine for Random's fixed
ziggurat tables, useless for per-call coordinates (it would mean recompiling the kernel per tile).
The real gap is narrower than "no constant memory": **no supported way to declare a runtime-written
`__constant__` array in a CUDA.jl kernel and address it from the host.** Kernel *parameters* do live
in constant bank 0 but cap at ~4 KB ≈ 256 points, only 2× the current tile.

**`CuTexture` was the fallback, and it fails on both branches:**

1. **Float64 textures do not exist.** `CuTextureArray(CuArray{Float64})` →
   `ArgumentError: CUDA does not support texture arrays for element type Float64`. Hardware limit
   (texture fetch is 32-bit), not a CUDA.jl gap. Float32 textures *do* work — verified with an exact
   fetch (`max|Δ| = 0`) in a real `@cuda` kernel. But the biggest SP2D win (2.58×) is the Float64
   case.
2. **For Float32, shared memory is not the binding constraint anyway.** Measured SP2D occupancy:

| | regs | block | shared | blk/SM (reg) | blk/SM (shared) | limiter |
|---|---|---|---|---|---|---|
| F32 32×16 | 40 | 1024 | 58,880 | **1** | 2 | REGISTERS |
| F32 50×50 | 40 | 1024 | 155,168 | **1** | 1 | REGISTERS |
| F64 32×16 | 58 | 512 | 71,936 | 2 | 2 | REGISTERS |
| F64 50×50 | 74 | 256 | 45,192 | 3 | 3 | REGISTERS |

Freeing the whole 32 KB staging via textures leaves 32×16 still at **1 block/SM** (40 regs × 1024
threads pins it), and 50×50 at 1 block (155 KB → 122 KB, still > 164/2). Zero occupancy gain.

**[M] This also corrects an inherited claim.** `OPTIMAL_KERNEL_DESIGN.md` states the dynamic
histogram "pins 1 block/SM, so threads/block IS occupancy" **[I]** — implying *shared* is the
constraint. On current code SP2D is **register**-limited in every configuration measured. Combined
with §8b (occupancy itself buys ≤3%), the entire "free up shared memory" family of optimizations is
dead for this workload.

### The decisive measurement — textures are SLOWER, timed **[M]**

The occupancy argument above is not sufficient on its own (§8b established this kernel is *not*
occupancy-bound, so "textures don't raise occupancy" does not prove they cannot help — a separate
cache path could still cut LSU traffic). So it was timed directly: two kernels, identical N-body pair
loop over `TILE=256`, `N=20000`, Float32; one stages the j-tile in shared, the other reads j from a
`CuTexture`.

| | time |
|---|---|
| j staged in **shared** | **4.024 ms** |
| j read from **texture** | 4.369 ms |
| ratio | **1.086 — texture 8.6% SLOWER** |

Identical results (max rel diff 4.1e-7, Float32 atomic ordering).

**[D] Mechanism.** In the N-body loop every lane reads the *same* address (`sxj[j]`), and a
shared-memory broadcast costs one cycle. A texture fetch goes through the texture cache at higher
latency. Replacing a free broadcast with a cached fetch can only lose.

### Calibration against CADISHI — we are NOT structurally behind **[M]**

The obvious objection: if constant memory does nothing, how did CADISHI reach ~495 bapps? Answer:
**that figure is for a pure distance histogram** — one *integer* atomic, no velocity increments, no
float accumulation. Measured here with a comparable kernel (thread owns `i` in registers, streams the
`j` tile from shared, `UInt32` shared histogram, 32 bins, N=20000, Float32, full N² ordered pairs):

| TILE | time | bapps |
|---|---|---|
| **256** | 1.086 ms | **368.3** |
| 512 | 1.134 | 352.7 |
| 1024 | 1.188 | 336.7 |
| 2048 | 1.238 | 323.0 |
| 4096 | 4.902 | 81.6 |

- **Our pure-distance rate is 368 bapps vs their ~495 on an older GPU generation — 1.35×, not a
  structural gap.** For comparable work we are near parity.
- **Larger tiles are monotonically WORSE.** This refutes the plausible story that constant memory's
  real value is a 20× bigger tile (5333 points vs 256): a bigger tile does not help *us* even when
  shared memory can hold it, so the 64 KB constant bank buys nothing on that axis either.
- Our real sp1d workload runs ~105 bapps. The **3.5× ratio to 368 is the cost of the physics**
  (`du`, `du·dx`, `|du|²`, 4 float atomics vs 1 integer atomic), consistent with the §2 cost model —
  not an implementation deficiency.

**[M] Correction to an earlier estimate in this session:** a first pass put our pure-distance rate at
~99 bapps and concluded we were 5× behind CADISHI. That kernel used a *float* atomic and parallelised
only over `lid <= ni`, so it was not a fair pure-distance comparison. The properly structured kernel
gives 368. Do not cite the 99 figure.

## 8. Register pressure is the binding constraint in 1D, and nobody has attacked it **[M]**

| | time | regs/thread | shared/block | blocks/SM (reg) | blocks/SM (shared) |
|---|---|---|---|---|---|
| Float64 | 5.75 ms | 48 | 14,848 B | **5** | 11 |
| Float32 | 4.77 ms | 46 | 7,680 B | **5** | 21 |

`ncu` confirms: `launch__occupancy_limit_registers = 5`, `_shared_mem = 6`, `_warps = 8`; achieved
`sm__warps_active = 61.3%` vs 62.5% predicted. `gpu__dram_throughput = 0.01%` — **not** memory-bound.
`sm__throughput = 76.5%`. Local memory 0, so no spilling.

Getting to 40 regs/thread → 6 blocks/SM (75% occupancy); to 32 → 8 blocks (100%). Float32 halves
shared and removes the Float64 16-slot bank floor (32 banks ÷ 2 words per Float64 = 16 independent
slots, forcing ≥2-way serialization for a 32-lane warp) yet gains only 1.2× — **confirming bank
conflicts are not the dominant term and register-limited occupancy is.**

## 8b. DEAD LEVER — occupancy is not the bottleneck **[M]**

§8 showed the sp1d kernel is register-limited, which *looks* like the obvious thing to attack. It is
not. Measured after the reroute, on the CUDA N-body kernel (N=20000):

| | regs | shared | blocks/SM | occupancy |
|---|---|---|---|---|
| Float32 | 38 | 11,776 B | 6 (reg-limited) | 75% |
| Float64 | 56 | 23,040 B | 4 (reg-limited) | 50% |

Capping registers with `maxregs` on the `@cuda` launch, sweeping the cap (Float64):

| maxregs | regs | blocks/SM | occupancy | time |
|---|---|---|---|---|
| default | 56 | 4 | 50% | 4.58 ms |
| 51 | 51 | 5 | 62.5% | 4.56 ms |
| 42 | 42 | 6 | 75% | 4.43 ms |
| 38 | 38 | 6 | 75% | 4.44 ms |
| 32 | 32 | 8 | 100% | 4.45 ms |

**`local memory / thread = 0` at every setting** — ptxas never spills, it rematerializes. So this is
a clean occupancy experiment with no spill confound, and **doubling occupancy from 50% to 100% buys
at most ~3%**; 75% → 100% buys nothing.

A careful 25-sample rerun of default vs `maxregs=42` is ambiguous — `min` 4.156 vs 4.306 (default
better), `median` 4.489 vs 4.333 and `mean` 4.444 vs 4.343 (cap better), against a default `std` of
0.145 ms. The metrics disagree and the effect is inside run-to-run variability.

**Conclusion: not applied, and register reduction is dead as a major lever.** The ~44% non-atomic
share of §2 is *not* recoverable through occupancy — it is latency/instruction-bound work. Do not
re-run this sweep; it costs GPU time and the answer is here.

## 8c. SP2D bin-count sweep: a 3× cliff, a hard failure, and a test-coverage hole **[M]**

Nobody had swept the bin count. Every prior SP2D measurement in this repo (mine included) used
32×16 or 50×50. Sweeping it (N=8000, full pipeline, A100):

| FT | bins | bapps | shared | kernel actually used |
|---|---|---|---|---|
| F32 | 32×16 | 7.6 | 58,880 | `_cuda_sf_2d_kernel` (CUDA fast path) |
| F32 | 64×64 | **5.6** | 45,192 | KA `sp2d_typeplane` (on-chip) |
| F32 | 100×100 | **1.9** | 4,200 | `sp2d_directpartition` ← **3× cliff** |
| F32 | 128×128 | 1.1 | 4,200 | `sp2d_directpartition` |
| F64 | 32×16 | 10.6 | 71,936 | CUDA fast path |
| F64 | 64×64 | **4.1** | 8,296 | `directpartition` ← cliff arrives earlier |

### BUG FIXED — SP2D threw for any `n_dist > 128` **[M]**

At ≥160×160 the call died with
`ArgumentError: naive single-pass 2D linear+linear requires vector value workspace`
(`ext/gpu/launch.jl`). Cause: `n_dist > SF_GPU_MAX_BINS` makes the tiled path ineligible, leaving
only the naive global kernel, which digitizes the value axis by binary search and demanded
`workspace.value_edges_sp2d_dev`. With no workspace supplied — the default — every large-bin call
failed outright. Fixed by reconstructing the edge vector from the linear plan's `first`/`last`
(the same allocate-on-demand pattern the general-bins routes already use).

Verified at 160×160 and 200×200 vs a CPU serial reference: **Σcounts exact**, Float64 sums to
1.7e-15 / 2.3e-15.

**Float32 shows 1.2–3.9% error there — and that is Float32, not the path.** Control:

| moment | CPU32 vs CPU64 | GPU32 vs CPU32 | GPU64 vs CPU64 |
|---|---|---|---|
| S2 | 1.67e-2 | 1.67e-2 | 4.5e-16 |
| L1T2 | 5.00e-2 | 3.87e-2 | 1.6e-15 |

The *CPU* in Float32 already differs from Float64 by 1.7–5.0%. A 160×160 histogram at N=1200 is
~28 pairs/cell with cancelling odd moments — beyond Float32's ~7 digits. **Use Float64 for large 2D
bin counts.** Always run this control before attributing a Float32 discrepancy to a code path.

### Why this was invisible: the test suite never goes there **[M]**

Audited `test/`: the largest 2D configuration tested anywhere is **60×60**
(`n_dist_bins`/`n_val_bins` ≤ 60); nothing exceeds ~128 bins on an axis except one *1D* over-cap
test. And **no test asserts which kernel or strategy runs.** So both the 3× routing cliff and the
outright failure sat in a regime that is neither tested nor benchmarked.

**Actionable gap:** add SP2D cases at 100×100 and 200×200 (Float64 for the sums check), and assert
the selected `SP2DAccumulationStrategy` mode where it matters, so a routing regression is visible.

### RESOLVED — route `:direct` to plain global atomics above a measured byte threshold **[M]**

Two candidate fixes were implemented and measured; **the simpler one won and the clever one was
reverted.**

1. **CUDA typeplane mode (implemented, measured, REVERTED).** Splitting the histogram into moment
   planes so the CUDA fast kernel stays on chip gave 100×100 → 3.2 bapps and 128×128 → 3.1, i.e.
   1.68–2.8× over `:direct`. But plain global atomics reach 4.4–4.8 there, so typeplane was ~1.5×
   *slower than doing nothing clever*. Reverted rather than left in the tree as a slower path;
   `_cuda_2d_pick_tile` keeps its original all-or-nothing test, now documented as deliberate.
2. **Routing (adopted).** `:direct` abandons the on-chip histogram for block-private global
   partitions plus a merge, so its cost grows with cells × tile-blocks, while plain global atomics get
   *cheaper* as more cells spread contention. Above `SP2D_GLOBAL_ATOMIC_HIST_BYTES = 340 KB`
   (`ext/gpu/sp2d_accumulation_strategy.jl`) `_launch_single_pass_2d!` routes past `:direct`.

| FT | bins | before | after | | kernel after |
|---|---|---|---|---|---|
| F32 | 32×16 | 7.6 | 7.6 | — | CUDA on-chip |
| F32 | 64×64 | 5.6 | 5.6 | — | `:typeplane` |
| F32 | 100×100 | 1.9 | **4.4** | 2.3× | naive |
| F32 | 128×128 | 1.1 | **4.8** | 4.4× | naive |
| F64 | 100×100 | 1.9 | **4.3** | 2.3× | naive |
| F64 | 128×128 | 1.1 | **3.5** | 3.2× | naive |
| both | 160×160 / 200×200 | ERROR | 4.4–7.1 | — | naive (§8c bug fix) |

**Why a threshold rather than "always skip `:direct`":** an unconditional skip made `:direct` — the
63-kernel template in `kernels_2d_direct.jl`, its partition buffers and merge kernels — unreachable,
and it overfits one device (the doc notes L40 has 100 KB and V100 96 KB of shared). The threshold
keeps `:direct` live below the crossover, where Float32 measured +11% for it, and captures every
measured win above.

**[M] Noise caution.** Float64 64×64 measured 4.1 and 3.1 bapps in two runs of the *same* code path
(253 KB < threshold, so untouched by this change). Run-to-run spread on that configuration is ~25%;
do not read differences below ~1.3× from single runs here.

### Superseded: "the CUDA 2D fast path has no typeplane mode" **[M]**

`_cuda_2d_pick_tile` requires the *whole* 6-moment histogram to fit
(`dynb = NMOM * n_dist * val_stride * (sizeof(FT)+4)`); otherwise it returns `TILE == 0` and declines
entirely, dropping to the KA path capped at 48 KB static and then to `directpartition`.
`grep -c typeplane ext/cuda/kernels_2d.jl` = **0**, while the KA path *does* have typeplane — and
typeplane is the mode beating directpartition 5.6 vs 1.9 bapps.

One plane at 64×64 Float32 is `1 × 64 × 65 × 8 = 32.5 KB`, plus 32 KB staging = 65 KB against a
163 KB opt-in — room for ~3 planes per pass. **Estimated ~3× in the 100×100–128×128 regime.** Kill
criterion: if a CUDA typeplane does not beat `directpartition`'s 1.9 bapps at 100×100, drop it.

## 8d. DEAD LEVER — `rsqrt` in place of `sqrt` + divide **[M]**

The flat per-pair path computes `r = sqrt(dot(dx,dx))` (`pair_frame`) and then `dot(δu, frame) / r`
(`pair_invariants`) — one `sqrt` plus one IEEE divide. `r` is needed only for the bin and `1/r` only
for `δu_L`, so both can in principle collapse into a single `rsqrt`: bin from `r²` (the CPU already
does this via `squared_digitize_plan`), and get `δu_L = dot(δu,dx) * inv_r`.

Measured with two otherwise-identical N-body kernels (TILE=256, N=20000, 32 bins, A100):

| | sqrt + divide | rsqrt | ratio |
|---|---|---|---|
| Float32 | 4.031 ms (99.2 bapps) | 4.150 ms (96.4) | **0.97× — slower** |
| Float64 | 6.644 ms (60.2 bapps) | 6.533 ms (61.2) | 1.02× — noise |

**No gain.** Consistent with §2: the ~44% non-atomic share is dominated by shared-memory loads and
loop overhead, not by the two transcendental ops, so removing one of them is invisible. Do not retry.

## 8e. `:typeplane` earns its complexity **[M]**

After the §7 routing change `:typeplane` survives in a narrow band, so it was worth asking whether it
still pays. At its operating point (Float32 64×64, 4096 cells) it measures **7.27 bapps vs 3.05 for
plain global atomics** — a 2.4× win. Keep it.

## 8f. Test coverage closed, and it immediately caught a bug **[M]**

§8c identified that nothing in the suite exercised 2D bin counts past 60×60 and that no test asserted
*which* kernel ran. Added to `test/test_gpu_sp2d_partitioned.jl`:

- **`GPU sp2d large bin counts`** — 100×100 and 200×200 against a CPU serial reference, no workspace
  (the configuration that used to raise `ArgumentError`), Float64 because Float32 disagrees with a
  Float64 reference by percent in this sparse regime even on the CPU (§8c).
- **`GPU sp2d strategy routing thresholds`** — asserts the byte crossover directly: 60×60 Float64 and
  80×80 Float32 below it, 100×100 Float64 and 128×128 Float32 above it.

**It found a real hole in the §7 fix on its first run.** The routing handled `GPUValueLinearShared`
but not the per-moment `GPUValueLinearCols` plan produced by an NTuple of value bins: above 128 bins
the global-atomic kernel has no method for it and the tiled fallback is ineligible, so it threw
`ArgumentError`. Root cause was the rule being **split across two places** — the value-plan builder
in `_gpu_run_single_pass_2d!` and the launcher in `_launch_single_pass_2d!` — which could disagree.
Both now consult one `_sp2d_prefers_global_atomics` predicate, and the plan builder produces
`GPUValueVectorCols` exactly when the launcher will route global.

## 9. Ordered exploration plan

Ordered by (expected gain) / (risk × effort). Each step states its own kill criterion.

1. ~~**Lever A, 6→4 moment atomics**~~ — **DONE (§3).** GPU 1.18×, CPU 1.31×, KA path done. (§3). Exact algebra, contained change, ~1.4× expected on an
   atomic-bound kernel. Kill if FP32 `T2`/`L1T2` error exceeds current tolerance on a turbulence
   field; fall back to FP64-only or partial reduction.
2. ~~**Per-regime A/B of non-batch → regime dispatcher**~~ — **DONE (§6, §6b).** 1.16–2.57×; 1D-individual deliberately excluded. (§6). No new kernels; 1.24–1.58× where it
   wins. Kill per regime where it loses — expected to lose on 1D-individual-fixed-x.
3. ~~**Register reduction to lift occupancy**~~ — **DEAD, measured (§8b).** 50% → 100% occupancy
   buys ≤3% and the metrics disagree at that scale. Not occupancy-bound.
4. **Spatial sort + run-length accumulation, then warp aggregation** (§4). Bounded by the ~2× of
   §2 for the atomic part — but it *also* unlocks tile culling, whose ceiling is set by the in-range
   pair fraction (0.75% at `r_max/L=0.05` **[M]**) and is therefore much larger when `r_max` is
   small. Subsumes the cell-list item. Kill if mean run length < 2 on a non-uniform field.
5. ~~**Constant/texture staging for SP2D**~~ — **DEAD, measured (§7).** Float64 textures do not
   exist in hardware, and Float32 SP2D is register-limited, not shared-limited, so freeing the
   staging gains no occupancy.

**Benchmark discipline** (learned the hard way, and worth repeating): always benchmark with the
production bin/value-plan types — a general binary-search fallback misrepresents the hot loop by
~3× **[I]**. Always compare Float32 *and* Float64; they are bound by different resources **[M]**.
And verify equivalence, not just speed, before reporting any win.
