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
`t = a + 5b = 0.00941` → **106.2 bapps vs 89.1 measured = ~1.19× predicted**, at B=64 Float32.
(An earlier estimate of ~1.4× assumed atomics dominated; §2 shows they are ~56%.)

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

## 6. Measured: the non-batch entry points run the slower structure **[M]**

Same workload, Float32, N=20000. Non-batch calls run the KA thread-owns-a-pair kernel; passing
`reshape(u, D, N, 1)` routes to the batch launcher, which reaches the CUDA N-body kernel.

| entry point | non-batch | batch B=1 (N-body) | ratio |
|---|---|---|---|
| single-pass 1D | 4.92 ms | 3.11 ms | **1.58×** |
| joint 2D | 4.38 ms | 2.94 ms | **1.49×** |
| single-pass 2D | 10.30 ms | 8.29 ms | **1.24×** |

Verified equivalent, not merely faster: worst relative difference across all six invariants is
**7.7e-14 (Float64)** and **3.3e-4 (Float32)** — consistent with a different summation order over
~4.8e7 in-range pairs, nothing more.

**Do not blanket-unify on this.** `OPTIMAL_KERNEL_DESIGN.md` records that 1D-individual **fixed-x**
is won by an older global-warp-replica design (147 bapps) over N-body (115), and that a unified
"one kernel for all regimes" regressed that regime ~28% **[I]**. The safe change is to route
non-batch into the *regime-dispatching* batch launcher, per regime, each gated on its own A/B —
including 1D-individual-fixed-x, which is the one regime expected to prefer the old kernel.

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

**Blocker:** CUDA.jl has no first-class constant-memory API — only a hacky `llvmcall` route and a
stalled draft PR. Alternatives worth evaluating: `CuTexture`/`CuDeviceTexture` (supported), `__ldg`
via `@Const` (already used, 55 sites), or kernel *parameters* (constant bank 0, but capped ~4 KB
≈ 256 points — only 2× our current tile).

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

## 9. Ordered exploration plan

Ordered by (expected gain) / (risk × effort). Each step states its own kill criterion.

1. **Lever A, 6→4 moment atomics** (§3). Exact algebra, contained change, ~1.4× expected on an
   atomic-bound kernel. Kill if FP32 `T2`/`L1T2` error exceeds current tolerance on a turbulence
   field; fall back to FP64-only or partial reduction.
2. **Per-regime A/B of non-batch → regime dispatcher** (§6). No new kernels; 1.24–1.58× where it
   wins. Kill per regime where it loses — expected to lose on 1D-individual-fixed-x.
3. **Register reduction to lift occupancy 5 → 6–8 blocks/SM** (§8). Inspect live ranges in the
   inner loop; the geometry frame and 6 moments are candidates. Kill if regs won't go below 40
   without spilling to local memory (currently 0 — spilling would be a net loss).
4. **Spatial sort + run-length accumulation, then warp aggregation** (§4). Bounded by the ~2× of
   §2 for the atomic part — but it *also* unlocks tile culling, whose ceiling is set by the in-range
   pair fraction (0.75% at `r_max/L=0.05` **[M]**) and is therefore much larger when `r_max` is
   small. Subsumes the cell-list item. Kill if mean run length < 2 on a non-uniform field.
5. **Constant/texture staging for SP2D only** (§7). Gated on CUDA.jl feasibility; do not attempt
   for 1D. Kill if `CuTexture` staging does not raise SP2D blocks/SM above 1.

**Benchmark discipline** (learned the hard way, and worth repeating): always benchmark with the
production bin/value-plan types — a general binary-search fallback misrepresents the hot loop by
~3× **[I]**. Always compare Float32 *and* Float64; they are bound by different resources **[M]**.
And verify equivalence, not just speed, before reporting any win.
