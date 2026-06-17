# Implement O(1) Flat Edge Binning and Clean Up Legacy Tuple Bins

This updated plan addresses the performance bottleneck in `StructureFunctions.jl` (`digitize` -> `searchsorted`) by implementing O(1) binning for flat edges. 

Additionally, because the codebase has no functional requirement for disjoint bins (and the digitize function already assumes contiguous bins), we will **entirely drop the legacy nested tuple bin representation** (`AbstractVector{<:Tuple}`) across the public API, internal loops, result objects, and tests. We will use flat vectors of boundaries (`AbstractVector`) everywhere.

## User Review Required

> [!IMPORTANT]
> - **Unified Flat Edge Design**: We will clean up the codebase by removing all tuple-based binning wrappers and conversions (such as `flat_bin_edges`). 
> - **Result Objects Constructor Verification**: `StructureFunction` and `StructureFunctionSumsAndCounts` will store flat edges of length `N + 1` for `N` values. Their constructors will be updated to validate `length(distance) == length(values) + 1` (instead of `length(distance) == length(values)`).
> - **Test Suite Migration**: All unit tests in the test suite currently passing tuples (e.g. `[(0.0, 10.0), (10.0, 20.0)]`) will be migrated to pass flat edge ranges/vectors (e.g. `[0.0, 10.0, 20.0]`).
> - **Reduced Annotation Footprint**: We will avoid unnecessary type restrictions (such as `<:Number`) on array parameters. We will rely on simple generic parameters like `AbstractVector` or type parameters `AbstractVector{T}` to keep the code clean and generic.

## Open Questions

None. The flat boundaries layout is cleaner, more performant, and has no legacy baggage.

## Proposed Changes

### StructureFunctions.jl Source Code

#### [NEW] [BinEdges.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/src/BinEdges.jl)
Define flat custom bin types and implement O(1) search methods:
- Define `AbstractBinEdges{T} <: AbstractVector{T}`.
- Define `BinEdges{T, ET <: AbstractVector{T}} <: AbstractBinEdges{T}` wrapping any sorted vector/range.
- Define `InfPaddedBinEdges{T, ET <: AbstractVector{T}} <: AbstractBinEdges{T}` wrapping any sorted vector/range, prepending `-Inf` and appending `+Inf`.
- Define `LogBinEdges{T, RT <: AbstractRange{T}, VT <: AbstractVector{T}} <: AbstractBinEdges{T}` wrapping log-spaced edges, storing a linear range of logs and a precomputed vector of actual boundary values to avoid `exp` evaluation on query time.
- Implement O(1) search methods:
  - `searchsortedfirst(v::BinEdges, x)` / `searchsortedfirst(v::BinEdges, x, o)`
  - `searchsortedfirst(v::InfPaddedBinEdges, x)` / `searchsortedfirst(v::InfPaddedBinEdges, x, o)`
  - `searchsortedfirst(v::LogBinEdges, x)` / `searchsortedfirst(v::LogBinEdges, x, o)`
  - `searchsortedlast(v::LogBinEdges, x)` / `searchsortedlast(v::LogBinEdges, x, o)`
  - `searchsorted(v::AbstractBinEdges, x, o)` returning `searchsortedfirst:searchsortedlast`

#### [MODIFY] [StructureFunctions.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/src/StructureFunctions.jl)
Include `BinEdges.jl` in the package module and export new types:
- `include("BinEdges.jl")`
- `export BinEdges, InfPaddedBinEdges, LogBinEdges, InfPaddedLogBinEdges`

#### [MODIFY] [StructureFunctionObjects.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/src/StructureFunctionObjects.jl)
Update result object constructors and methods to check/validate flat edges layout:
- In `StructureFunction` constructor: change check to `length(distance) == length(values) + 1`.
- In `StructureFunctionSumsAndCounts` constructor: change check to `length(distance) == length(sums) + 1`.
- In `StructureFunction2D` constructor: change check to `size(sums) == (length(distance_bins) - 1, length(value_bins) - 1)`.
- Update `marginalize` to correctly extract the 1D structure function using the flat edges.

#### [MODIFY] [Calculations.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/src/Calculations.jl)
Remove all tuple conversions:
- Remove `flat_bin_edges` entirely.
- Change all signatures that accepted `AbstractVector{<:Tuple}` or `AbstractVector` to accept generic `AbstractVector`.
- Remove step that converts flat ranges to tuple vectors inside `calculate_structure_function`.
- Ensure all loops directly index `distance_bins` and `value_bins` using O(1) indexing and digitize dispatch.

### Test Suite

#### [NEW] [test_bin_edges.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/test/test_bin_edges.jl)
Add comprehensive correctness and performance unit tests for `BinEdges`, `InfPaddedBinEdges`, and `LogBinEdges`.

#### [MODIFY] [runtests.jl](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/test/runtests.jl)
Include `test_bin_edges.jl` in the test runner.

#### [MODIFY] all test files in [test/](file:///home/jbenjami/Code/jbphyswx/StructureFunctions.jl/test)
Migrate all tuple definitions (e.g. `distance_bins = [(0.0, 10000.0), ...]`) to flat edge ranges/vectors (e.g. `distance_bins = [0.0, 10000.0, ...]`).

## Verification Plan

### Automated Tests
Run Julia unit tests to verify baseline compatibility, type stability, JET inference, and correctness:
```bash
julia --project=@. -e "using Pkg; Pkg.test()"
```

### Manual Verification
Run the python benchmark script `benchmark_sf.py` (which runs julia backend) to verify throughput and speedup.
