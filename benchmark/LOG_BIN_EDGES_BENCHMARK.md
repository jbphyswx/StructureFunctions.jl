# LogBinEdges digitize benchmark

Run: `julia --project=. benchmark/benchmark_log_bin_edges_all.jl`  
Log: `test/debug/log_bin_edges_all_benchmark.log`

Timing: 10_000 random queries in `[1, 1000]`, 5 repetitions, best ns/query.  
Errors: larger verify set (random + edge ± eps + out-of-range).

## Unified design (one path)

All log-spaced bins digitize via **`log(q)` + `LinearBinEdges` FMA** on the log grid (`floor(t)+1` + ULP correction). No exponent LUT.

| ID | What it is |
|----|------------|
| P1 | Binary search on physical `Vector` (baseline) |
| P2 | `LogBinEdges(phys_vec)` — builds log grid, same digitize as P5 |
| P2b | `LogBinEdges_from_log_edges` — log grid direct |
| P5 | `log(q)` + `LinearBinEdges` reference (explicit) |
| P6d | `floor(t)+1` **without** correction — regression guard |

## Error columns

- **err_phys** — `searchsortedfirst(exp.(log_edges), q)` — physical-vector binary search (not package spec)
- **err_lin** — `searchsortedfirst(LinearBinEdges(log_edges), log(q))` — **package spec** (P2/P2b/P5 must be 0)

## Expected (F64 N=1000)

| Path | ns/query | err_lin |
|------|----------|---------|
| P2 / P2b / P5 | ~11 | 0 |
| P1 physical vector | ~46 | — |
| P6d no correction | ~9 | ~81% wrong |

`log(q)` ~6 ns; FMA+corr ~2.5 ns on precomputed `lx`.

See `docs/UNIFORM_BIN_DIGITIZE.md` for derivation.
