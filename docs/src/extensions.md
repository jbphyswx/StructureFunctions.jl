# Extensions

StructureFunctions.jl keeps optional integrations behind Julia package extensions. The core package should stay focused on array-based structure-function calculations; extensions only add execution backends or visualization helpers.

## Available Extensions

### OhMyThreads

Loading `OhMyThreads.jl` activates threaded CPU methods used by `ThreadedBackend()`.

```julia
using StructureFunctions
using OhMyThreads

result = calculate_structure_function(sf, x, u, bins; backend = ThreadedBackend())
```

### Distributed

Loading `Distributed` activates the distributed CPU methods used by `DistributedBackend()`.

```julia
using StructureFunctions
using Distributed

addprocs(4)
result = calculate_structure_function(sf, x, u, bins; backend = DistributedBackend())
```

### KernelAbstractions

Loading `KernelAbstractions.jl` activates GPU and KA.CPU methods used by `GPUBackend`.

```julia
using StructureFunctions
using KernelAbstractions

backend = GPUBackend(KernelAbstractions.CPU())
result = calculate_structure_function(sf, x, u, bins; backend)
```

Use CUDA.jl or another KernelAbstractions backend package to run on actual accelerator hardware.

### CairoMakie

Loading `CairoMakie.jl` activates plotting helpers, if those helpers are used by downstream code.

```julia
using StructureFunctions
using CairoMakie
```

## Not Provided

StructureFunctions.jl does not provide file-format loaders. Pass arrays directly to the calculation APIs:

```julia
calculate_structure_function(sf, x, u, bins)
```

Application-specific I/O, preprocessing, and metadata handling should live outside this package.

## Project.toml Entries

The package currently declares only the optional dependencies needed by the remaining extensions:

```toml
[weakdeps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"

[extensions]
StructureFunctionsCairoMakieExt = ["CairoMakie"]
StructureFunctionsDistributedExt = ["Distributed"]
StructureFunctionsKernelAbstractionsExt = ["KernelAbstractions"]
StructureFunctionsOhMyThreadsExt = ["OhMyThreads"]
```

## Related Topics

- [Backends](backends.md)
- [GPU](gpu.md)
- [Architecture](architecture.md)
