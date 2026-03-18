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

### Result Container

```julia
struct StructureFunction{T}
    distance::Vector{T}          # Bin centers
    structure_function::Matrix{T} # S(distance, order)
    sums::Matrix{T}              # Numerator sums
    counts::Vector{Int64}        # Counts per bin
end
```

Stores **both raw and processed data** so users can customize post-processing.

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
