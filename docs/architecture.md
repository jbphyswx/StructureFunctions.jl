# Architecture: Design & Implementation

This document explains how StructureFunctions.jl is organized internally and how computations are dispatched.

## Table of Contents
- [Module Organization](#module-organization)
- [Type Hierarchy](#type-hierarchy)
- [Backend Dispatch](#backend-dispatch)
- [Extension System](#extension-system)
- [Code Layout](#code-layout)

## Module Organization

### Core Module: StructureFunctions

The main module (`src/StructureFunctions.jl`) defines:
- **Type definitions**: `AbstractExecutionBackend`, `SerialBackend`, `ThreadedBackend`, etc.
- **Core functions**: `calculate_structure_function`, dispatcher methods
- **Result container**: `StructureFunction` type for storing results

### Architecture Pattern: Operator × Container

StructureFunctions.jl uses a **operator composition** pattern:

```
Data (x, u)
    ↓
[StructureFunction Operator] ← Type specifies WHICH calculation
    ↓
[Execution Backend] ← Type specifies HOW to compute
    ↓
Result Container ← Stores sums, counts, structure functions
```

Example:
```julia
# Operator: "Compute 2nd-order SF"
operator = FullVectorStructureFunction{Float64}(order=2)

# Backend: "Use 4 threads"
backend = ThreadedBackend()

# Call dispatcher with both
SF_result = calculate_structure_function(operator, x, u, bins; backend)
```

The **operator type** determines *what calculation* to perform (which SF variant, which order). The **backend type** determines *how* to execute it (serial, threaded, GPU, etc.).

---

## Type Hierarchy

### Execution Backends

All backends inherit from `AbstractExecutionBackend`:

```
AbstractExecutionBackend (abstract)
├── SerialBackend
├── ThreadedBackend
├── DistributedBackend
├── GPUBackend{B}  [parametric in device backend]
└── AutoBackend
```

**Key property**: Each backend type is **singleton-like** — zero memory overhead, pure dispatch. Example:
```julia
serial = SerialBackend()        # Type ≈ Singleton
threaded = ThreadedBackend()    # Type ≈ Singleton
```

### Structure Function Operators

Operators describe *which* calculation variant:

```
AbstractStructureFunctionType (abstract)
├── ProjectedStructureFunction{T}
│   └── Computes SF(r) at each distance for a single projection
│       (e.g., longitudinal, transverse, vertical)
└── FullVectorStructureFunction{T}
    └── Computes multi-dimensional SFs using full velocity vectors
        (e.g., 2D/3D anisotropic flows)
```

Each operator stores:
- **Vector type** (Float32, Float64) via type parameter
- **Order** (n=2, 3, 4, ...) — which structure function order
- **Projection** (if applicable) — which component to analyze

### Bin Edges (AbstractBinEdges)

To eliminate the $O(\log B)$ binary search overhead in distance binning, StructureFunctions.jl provides custom, fast, zero-allocation collections subtyping `AbstractBinEdges{T}`:

```
AbstractBinEdges (abstract)
├── BinEdges           [fallback wrapper for standard vectors]
├── LinearBinEdges     [O(1) FMA-based search for uniform ranges]
├── LogBinEdges        [O(1) Exponent LUT Hybrid search for log ranges]
└── InfPaddedBinEdges  [wrapper to append/prepend ±∞ boundaries]
```

These types implement custom `Base.searchsortedfirst` overrides, enabling highly efficient $O(1)$-like bin lookups within core calculations.

#### Algorithmic Design

##### 1. Linear Binning via Fused Multiply-Add (FMA)
When bin edges are uniformly spaced (a range), a standard binary search takes $O(\log B)$ steps. Instead, `LinearBinEdges` performs a constant-time $O(1)$ mapping from value to index:
$$\text{index}(x) = \text{round}\left(\text{Int}, x \cdot \text{inv\_step} + \text{offset}\right)$$
where $\text{inv\_step} = 1/\delta$ and $\text{offset} = 1 - v_1/\delta$. By evaluating this via a Fused Multiply-Add (`muladd`) instruction, the CPU executes it in a single cycle. A subsequent $O(1)$ floating-point boundary check corrects any potential 1-ULP precision mismatch at bin edges, ensuring 100% numerical parity with standard binary search in only **~3 ns** (a **15x+ speedup**).

##### 2. Log Binning via Exponent Lookup Tables (LUT)
Logarithmic binning normally requires evaluating $\ln(x)$ to map the value to linear space, but the hardware `log` instruction is highly latent (20-40 CPU cycles). To bypass this, `LogBinEdges` extracts the binary floating-point exponent of the query value using `exponent(x)`. This operation is an IEEE 754 bit-mask and shift, taking **< 0.5 ns**. 

During construction, a Lookup Table (LUT) is created to map each binary exponent (octave) to the first index in the bin edges vector that intersects with that octave. When searching:
1. Extract exponent $e = \text{exponent}(x)$.
2. Query the precomputed LUT to retrieve bounds `idx_start` and `idx_end` for that octave, restricting the search range.
3. Perform a hybrid search: if the restricted subrange contains $\le 8$ elements, use a fast cache-friendly linear scan; otherwise, run a binary search restricted to that subrange.
This hybrid strategy bypasses `log(x)` completely, reducing lookups to **~5-8 ns** (a **5x+ speedup**).

##### 3. Out-Of-Bounds Handling via Virtual Padding
Calculations need to determine if a point pair's separation falls within the bounds of the bin edges. Rather than introducing branching code inside inner loops, `InfPaddedBinEdges` virtually prepends $-\infty$ (or `typemin(T)`) and appends $+\infty$ (or `typemax(T)`) to any existing `AbstractBinEdges` collection. Out-of-bounds inputs automatically fall into the boundary pads in $O(1)$ time without copying or allocating additional memory.

### Result Containers

StructureFunctions.jl decouples raw accumulation, processed 1D structure functions, and 2D joint-probability binning into separate parametric result types inheriting from `AbstractStructureFunction`:

1. **`StructureFunction`**: Stores the final processed structure function values.
```julia
struct StructureFunction{FT, OT, BT, VT} <: AbstractStructureFunction
    operator::OT                   # AbstractStructureFunctionType
    distance_bins::BT              # AbstractVector of (r_min, r_max)
    values::VT                     # AbstractVector{FT} — computed SF
    order::Int                     # 1, 2, 3, ...
end
```

2. **`StructureFunctionSumsAndCounts`**: Stores exact computed sums and point counts per bin. Ideal for distributed or chunked temporal aggregation.
```julia
struct StructureFunctionSumsAndCounts{FT, OT, BT, VT} <: AbstractStructureFunction
    operator::OT
    distance_bins::BT
    sums::VT                       # Exact computed SF value sums
    counts::VT                     # Integer counts of contributing pairs
end
```

3. **`StructureFunction2D`**: Stores the 2D joint-probability binning grid (separation distance $r$ vs. SF value $v$).
```julia
struct StructureFunction2D{FT, OT, BT, VT, MT} <: AbstractStructureFunction
    operator::OT
    distance_bins::BT
    value_bins::VT                 # Value increment bin edges
    sums::MT                       # 2D matrix of exact sums (distance x value)
    counts::MT                     # 2D matrix of contribution counts
end
```

All result containers support basic `Base` algebraic operations (like `+` and `+=`) to allow seamless aggregation across distributed processes or temporal timesteps.

---

## Backend Dispatch

### Dispatch Flow

When you call `calculate_structure_function(operator, x, u, bins; backend)`:

1. **Type signature selected** based on `backend` type
2. **Preparation phase** (same for all backends):
   - Validate inputs
   - Allocate result container
   - Set up spatial binning
3. **Execution phase** (backend-specific):
   - `SerialBackend`: Single loop over points
   - `ThreadedBackend`: Multi-threaded loop via OhMyThreads
   - `DistributedBackend`: Distribute over processes
   - `GPUBackend`: Launch kernels
   - `AutoBackend`: Detect available resources → select best backend
4. **Reduction phase** (same for all backends):
   - Finalize sums and normalize
   - Store in result container

### Code Structure

```
src/
├── Calculations.jl          # Core calculation logic (backend-agnostic)
├── StructureFunctionTypes.jl # Operator type definitions
├── HelperFunctions.jl        # Utilities (binning, normalization)
└── Backends.jl              # Backend type definitions

ext/
├── ThreadedBackend.jl       # OhMyThreads integration
├── DistributedBackend.jl    # Distributed.jl integration
├── GPUBackend.jl            # KernelAbstractions integration
├── JLD2Ext.jl              # JLD2 file I/O
├── NetCDFExt.jl             # NetCDF file I/O
└── ... (more extensions)
```

### Example: ThreadedBackend Dispatch

When `backend=ThreadedBackend()` is passed:

```julia
# Simplified view of internal dispatcher
calculate_structure_function(op::StructureFunctionType, 
                             x, u, bins;
                             backend::ThreadedBackend) = begin
    # Setup (shared)
    result = StructureFunction(...)
    
    # Execution (ThreadedBackend-specific)
    # Uses OhMyThreads.tmapreduce to parallelize point-pair iteration
    compute_threaded!(result, x, u, bins)
    
    # Finalize (shared)
    normalize!(result)
    
    return result
end
```

This **method specialization** ensures:
- ✅ No runtime overhead choosing between backends
- ✅ Each backend can use its best algorithm
- ✅ Type-stable dispatch

---

## Extension System

### Lazy Loading via Extensions

Optional dependencies are loaded **only when needed** via Julia's extension mechanism:

```toml
[weakdeps]
OhMyThreads = "67456a42-ebe4-4781-8ad1-67f7eda8d8f7"
Distributed = "8ba89e20-285c-5519-8a0c-887f00cd4b76"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1f7f1"

[extensions]
OhMyThreadsExt = "OhMyThreads"
DistributedExt = "Distributed"
GPUExt = "KernelAbstractions"
```

**Benefits**:
- Users who don't use ThreadedBackend pay zero cost (no OhMyThreads load time)
- GPU users can optionally install KernelAbstractions
- Fresh Julia session starts fast (no big dependency tree by default)

### Adding a New Extension

To add support for a new backend (e.g., `CUDABackend`):

1. **Add weakdep** in Project.toml:
   ```toml
   CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
   ```

2. **Create extension** `ext/CUDAExt.jl`:
   ```julia
   module CUDAExt
   
   using StructureFunctions
   using CUDA
   
   struct CUDABackend end
   
   function calculate_structure_function(op, x, u, bins;
                                        backend::CUDABackend)
       # CUDA-specific dispatch
   end
   
   end  # module
   ```

3. **Publish** as part of release

### Example: Using NetCDFExt

If you have climate data in NetCDF format:

```julia
using StructureFunctions
using NetCDF

# Extension auto-loads when NetCDF is available
# Registers a convenience method
SF = calculate_structure_function("path/to/data.nc",
                                 backend=ThreadedBackend())
```

---

## Code Layout

### Key Files

| File | Purpose |
|------|---------|
| `src/Calculations.jl` | Main dispatcher; backend-agnostic logic |
| `src/StructureFunctionTypes.jl` | Operator type definitions |
| `src/HelperFunctions.jl` | Binning, distance metrics, utils |
| `src/Backends.jl` | Backend type definitions |
| `ext/ThreadedBackend.jl` | OhMyThreads integration |
| `ext/GPUExt.jl` | KernelAbstractions + GPU kernels |
| `src/__init__.jl` | Exports public types/functions |

### Import Strategy

```julia
# src/StructureFunctions.jl (main module)

# Public exports
export SerialBackend, ThreadedBackend, DistributedBackend,
       GPUBackend, AutoBackend,
       calculate_structure_function,
       StructureFunction

# Dependencies
using LinearAlgebra
using Distances
using ProgressMeter
using StaticArrays

# No imports of optional dependencies (those are extensions)
```

### Public vs Internal

**Public API** (safe to use, won't change):
- `calculate_structure_function` function
- All `*Backend` types
- `StructureFunction` container
- Exported operator types

**Internal** (subject to change):
- Helper functions in `HelperFunctions.jl` marked `@doc hide`
- Kernel implementations in extensions
- Intermediate data structures

---

## Design Principles

1. **Type Dispatch**: Use Julia's type system, not string dispatch
   - ✅ Static overhead elimination
   - ✅ Runtime type safety
   - ✅ IDE autocompletion

2. **Zero-Cost Abstraction**: Backend dispatch adds no runtime cost
   - Single method per backend type
   - Compiler resolves at dispatch time
   - No runtime branching

3. **Extensibility**: Users can add custom backends
   - Define new `Backend <: AbstractExecutionBackend` type
   - Define `calculate_structure_function` method for it
   - Works instantly (static dispatch)

4. **Separation of Concerns**:
   - Operators describe *what* to compute (decoupled from backend)
   - Backends describe *how* to compute (decoupled from operator)
   - Result container is pure data (independent of both)

---

## Related Topics

- [Theory](theory.md): Mathematical foundations
- [Backends](backends.md): When to use each backend
- [Real Data](real_data.md): File I/O and extensions
- [Examples](../examples/README.md): Worked code examples
