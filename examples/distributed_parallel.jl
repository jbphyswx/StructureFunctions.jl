# Partition pair work across local Julia workers and reduce their histograms.
include("resources.jl")
using Distributed: Distributed
using Random: Random
using StructureFunctions: Calculations as C, StructureFunctionTypes as T, StructureFunctionSumsAndCounts
using ComputationalBackends: DistributedBackend, SerialBackend

function distributed_example(; n=ExampleResources.points(), workers=2)
    workers >= 1 || throw(ArgumentError("workers must be positive"))
    existing = filter(!=(myid()), Distributed.workers())
    total = max(length(existing), workers)
    ExampleResources.require_allocation(; cpus=total + 1)
    project = realpath(dirname(Base.active_project()))
    for worker in existing
        dirname(realpath(remotecall_fetch(Base.active_project, worker))) == project ||
            error("Existing worker $worker uses a different project")
    end
    created = Int[]
    try
        append!(created, addprocs(max(0, workers - length(existing));
            exeflags=`--project=$project --threads=1 --startup-file=no`,
            env=["OPENBLAS_NUM_THREADS"=>"1", "MKL_NUM_THREADS"=>"1"]))
        for worker in filter(!=(myid()), Distributed.workers())
            remotecall_wait(Core.eval, worker, Main,
                :(using StructureFunctions, ComputationalBackends, LinearAlgebra))
            remotecall_wait(Core.eval, worker, Main, :(LinearAlgebra.BLAS.set_num_threads(1)))
        end
        rng = Random.MersenneTwister(14)
        x, u = Random.rand(rng, 2, n), Random.randn(rng, 2, n)
        bins = range(0.0, 1.5; length=7)
        calculate(backend) = C.calculate_structure_function(T.S2SFType(), x, u, bins,
            StructureFunctionSumsAndCounts; backend)
        result = calculate(DistributedBackend(SerialBackend()))
        reference = calculate(SerialBackend())
        @assert result.counts == reference.counts
        @assert result.sums ≈ reference.sums
        println("Matched ", sum(result.counts), " pairs across ", total, " workers")
        return result
    finally
        isempty(created) || rmprocs(created)
    end
end
result = distributed_example()
