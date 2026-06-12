# Pwd-independent entry for gpu/ scripts. Load once per REPL (`--project=gpu`):
#
#   include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
#   include_gpu("diagnose_sums.jl")
#   include_gpu("benchmark_prototypes.jl")

using Pkg: pkgdir
using StructureFunctions: StructureFunctions

const SF_REPO = pkgdir(StructureFunctions)
const SF_GPU_DIR = joinpath(SF_REPO, "gpu")

"""Include any script under `gpu/`; path does not depend on `pwd`."""
function include_gpu(name::AbstractString)
    path = joinpath(SF_GPU_DIR, name)
    isfile(path) || error("gpu script not found: $path")
    include(path)
    return nothing
end
