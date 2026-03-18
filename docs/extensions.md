# Extensions: Lazy Loading & Customization

This document explains how StructureFunctions.jl uses Julia's extension system and how to develop custom extensions.

## Table of Contents
- [Extension System Overview](#extension-system-overview)
- [Available Extensions](#available-extensions)
- [Loading Extensions](#loading-extensions)
- [Developing Custom Extensions](#developing-custom-extensions)
- [Extension Lifecycle](#extension-lifecycle)

---

## Extension System Overview

### What Are Extensions?

Extensions are optional code modules that integrate StructureFunctions.jl with external libraries. They're loaded **only when needed**, keeping the core library lightweight.

### Benefits

1. **Fast startup**: Core library loads instantly (no heavy dependencies)
2. **Flexibility**: Users choose which features they need
3. **Clear dependencies**: No surprise required packages
4. **Clean code organization**: Separate concerns (backends, I/O, etc.)

### How Extensions Work

Extension registration in [Project.toml](../Project.toml):

```toml
[weakdeps]
OhMyThreads = "67456a42-ebe4-4781-8ad1-67f7eda8d8f7"
Distributed = "8ba89e20-285c-5519-8a0c-887f00cd4b76"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1f7f1"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
NetCDF = "30363a11-e695-5f81-9200-c891c9c60a12"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3986"

[extensions]
OhMyThreadsExt = "OhMyThreads"        # Optional dependency
DistributedExt = "Distributed"         # Optional dependency
GPUExt = "KernelAbstractions"          # Optional dependency
CUDAExt = "CUDA"                       # Optional dependency
NetCDFExt = "NetCDF"                   # Optional dependency
JLD2Ext = "JLD2"                       # Optional dependency
```

**Key point**: Weak dependency = optional; extension loads automatically when package is imported.

---

## Available Extensions

### Backend Extensions

#### OhMyThreadsExt (ThreadedBackend)

Enables multi-threaded execution via OhMyThreads.jl.

**Status**: Included in StructureFunctions.jl  
**When loaded**: Automatically when `using OhMyThreads`  
**Provides**: `ThreadedBackend()` type and dispatcher  

```julia
using StructureFunctions
using OhMyThreads  # Triggers extension loading

backend = ThreadedBackend()
result = calculate_structure_function(...; backend)
```

#### DistributedExt (DistributedBackend)

Enables distributed-memory execution (multi-process, multi-machine).

**Status**: Included in StructureFunctions.jl  
**When loaded**: Automatically when `using Distributed`  
**Provides**: `DistributedBackend()` type and dispatcher  

```julia
using StructureFunctions
using Distributed
addprocs(4)

backend = DistributedBackend()
result = calculate_structure_function(...; backend)
```

#### GPUExt (GPUBackend)

Enables portable GPU execution via KernelAbstractions.jl.

**Status**: Included in StructureFunctions.jl  
**When loaded**: Automatically when `using KernelAbstractions`  
**Provides**: `GPUBackend()` type and CUDA/ROCm/Metal kernel implementations  

```julia
using StructureFunctions
using CUDA  # Triggers extension loading

backend = GPUBackend()
result = calculate_structure_function(...; backend)
```

### I/O Extensions

#### JLD2Ext

Read/write binary Julia data files in efficient HDF5 format.

**Status**: Included in StructureFunctions.jl  
**When loaded**: Automatically when `using JLD2`  
**Provides**: Helper functions for saving/loading `StructureFunction` results  

```julia
using StructureFunctions
using JLD2

# Calculate
result = calculate_structure_function(...)

# Save to disk
save_structure_function("results.jld2", result)

# Load
loaded_result = load_structure_function("results.jld2")
```

#### NetCDFExt

Read climate/geophysical data in NetCDF format (HDF5-based).

**Status**: Included in StructureFunctions.jl  
**When loaded**: Automatically when `using NetCDF`  
**Provides**: Convenience function to load velocity data from NetCDF  

```julia
using StructureFunctions
using NetCDF

# Load temperature and wind from climate model output
data = load_netcdf_data("model_output.nc", 
                       variables=["temperature", "u", "v"])

# Automatic file → calculation pipeline
result = calculate_structure_function(data.x, data.u, bins;
                                     backend=AutoBackend())
```

#### ZarrExt (Future)

Read cloud-native array format (Zarr protocol).

**Status**: Planned for v0.4  
**When available**: Will auto-load with `using Zarr`  
**Use case**: Distributed cloud storage (AWS S3, Google Cloud Storage)  

```julia
# Planned (v0.4):
using StructureFunctions
using Zarr

data = open_zarr("s3://bucket/climate_data.zarr")
result = calculate_structure_function(data, backend=DistributedBackend())
```

---

## Loading Extensions

### Automatic Loading

Extensions load automatically when their weak dependency is imported:

```julia
# This automatically loads OhMyThreadsExt
using StructureFunctions
using OhMyThreads  # ← Triggers extension

backend = ThreadedBackend()  # Now available!
```

### Manual Verification

Check which extensions are loaded:

```julia
using StructureFunctions

# List all loaded extensions
using Pkg
Pkg.status()  # Shows which [extras] are loaded

# Or in Julia directly
@show StructureFunctions.ThreadedBackend  # ✓ Exists if loaded
@show StructureFunctions.DistributedBackend  # ✓ Exists if loaded
```

### Optional vs Required

**Optional** (user chooses):
- `OhMyThreads` — Can use SerialBackend instead
- `Distributed` — Can use ThreadedBackend instead
- `KernelAbstractions` — Can use CPU backends instead
- `NetCDF`, `JLD2`, `Zarr` — Can load data manually

**Required** (StructureFunctions.jl can't work without):
- `LinearAlgebra` — Core Julia library
- `Distances.jl` — Distance metric definitions
- `StaticArrays.jl` — Type-stable small arrays

---

## Developing Custom Extensions

### Example: Custom Backend

Suppose you want to add a **FPGABackend** for custom hardware.

### Step 1: Create Extension Module

Create file `ext/FPGAExt.jl`:

```julia
# Ensure extension only loads when FPGA.jl is imported
module FPGAExt

using StructureFunctions
using FPGA  # The weak dependency

# Export new backend type (will be available as StructureFunctions.FPGABackend)
# No `using StructureFunctions: FPGABackend` needed; it's in the module

struct FPGABackend <: StructureFunctions.AbstractExecutionBackend
    device_id::Int = 0  # Which FPGA to use
end

# Define the dispatcher
function StructureFunctions.calculate_structure_function(
    operator::StructureFunctions.StructureFunctionType,
    x::AbstractMatrix,
    u::AbstractMatrix,
    bins::AbstractVector;
    backend::FPGABackend,
    kwargs...
)
    # Your FPGA-specific implementation
    
    # Example pseudocode:
    fpga_kernel = compile_for_fpga(operator)
    result = StructureFunctions.StructureFunction(...)
    
    # Call hardware
    compute_on_fpga!(result, fpga_kernel, x, u, bins, backend.device_id)
    
    return result
end

end  # module FPGAExt
```

### Step 2: Update Project.toml

In the package root [Project.toml](../Project.toml):

```toml
[weakdeps]
FPGA = "12345678-abcd-ef01-2345-6789abcdef01"

[extensions]
FPGAExt = "FPGA"
```

### Step 3: Test the Extension

```julia
# In an environment where FPGA.jl is installed:
using StructureFunctions
using FPGA

# Extension auto-loads
backend = FPGABackend(device_id=0)

result = calculate_structure_function(operator, x, u, bins;
                                     backend=backend)
```

### Step 4: Document

Add documentation to [docs/extensions.md](extensions.md):

```markdown
#### FPGAExt (Custom Hardware)

Your description here.
```

---

## Extension Lifecycle

### Loading Sequence

1. **Julia starts**
   ```
   julia -e "using StructureFunctions"
   → Main module loads
   → Core types exported
   → Extensions NOT loaded yet (no weakdeps imported)
   ```

2. **User imports weak dependency**
   ```julia
   using StructureFunctions
   using OhMyThreads  # ← Triggers package loading
   → Julia detects OhMyThreads is available
   → OhMyThreadsExt automatically loads
   → `ThreadedBackend()` now available
   ```

3. **Calling code**
   ```julia
   ThreadedBackend()  # Type resolution finds ThreadedBackend
   # Dispatcher method selected at compile-time
   ```

### Memory & Performance

**With extended dependencies loaded**:
- OhMyThreads: +5 MB memory, ~50 ms load time
- Distributed: +15 MB memory, ~100 ms load time
- KernelAbstractions: +20 MB memory, ~200 ms load time
- NetCDF: +8 MB memory, +100 ms load time

**Without any extensions**:
- StructureFunctions: ~2 MB memory, ~5 ms load time

This is why **lazy loading matters**.

---

## Best Practices

### For Users

1. **Load only what you need**
   ```julia
   # ✓ Good: Only load GPU stuff if using GPU
   if use_gpu
       using CUDA
   end
   
   # ✗ Bad: Always load everything
   using StructureFunctions, OhMyThreads, Distributed, CUDA, NetCDF
   ```

2. **Use AutoBackend when possible**
   ```julia
   # ✓ Good: Let library adapt to environment
   result = calculate_structure_function(...; backend=AutoBackend())
   
   # ✗ Bad: Hard-code backend
   result = calculate_structure_function(...; backend=ThreadedBackend())
   ```

### For Extension Developers

1. **Use clear module names**
   ```julia
   # ✓ Good: Clear which dependency
   module OhMyThreadsExt
   module FluxExt
   
   # ✗ Bad: Too generic
   module ThreadingExt
   module MLExt
   ```

2. **Preserve type stability**
   ```julia
   # ✓ Good: Type stable dispatch
   function calculate_structure_function(...; backend::MyBackend)
       # Type known at compile-time
   end
   
   # ✗ Bad: Loses type info
   function calculate_structure_function(...; backend_name::String)
       # Requires runtime dispatch
   end
   ```

3. **Document prerequisites**
   ```
   ## MyExt
   
   **Requires**: YourPackage.jl v1.2+
   **Status**: Stable / Experimental / Deprecated
   **Example**:
   ```julia
   using StructureFunctions
   using YourPackage  # Triggers auto-load
   ...
   ```
   ```

---

## Related Topics

- [Architecture](architecture.md): Extension system design
- [Backends](backends.md): When to use each execution backend
- [Real Data](real_data.md): Using I/O extensions
- [Examples](../examples/README.md): Practical workflows
