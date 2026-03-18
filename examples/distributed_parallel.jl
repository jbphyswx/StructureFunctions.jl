"""
    distributed_parallel.jl

Distributed backend example with strict qualified imports.

Run from package root:
    julia --project=examples examples/distributed_parallel.jl 4
"""

using Distributed: Distributed
using StructureFunctions: StructureFunctions as SF

n_workers = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2
if Distributed.nworkers() < n_workers
    Distributed.addprocs(n_workers - Distributed.nworkers())
end

Distributed.@everywhere using StructureFunctions: StructureFunctions as SF

N = 100_000
x = randn(N, 2)
u = randn(N, 2)
bins = collect(10.0:20.0:1000.0)
operator = SF.FullVectorStructureFunctionType{Float64}(order = 2)

result = @time SF.calculate_structure_function(
    operator,
    x,
    u,
    bins;
    backend = SF.DistributedBackend(),
    show_progress = false,
    verbose = false,
)

println("Workers: $(Distributed.nworkers())")
println("Total pair counts: $(sum(result.counts))")

Distributed.rmprocs(Distributed.workers())
