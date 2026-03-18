"""
    threaded_calculation.jl

Threaded backend example with strict qualified imports.

Run from package root:
    JULIA_NUM_THREADS=4 julia --project=examples examples/threaded_calculation.jl
"""

using StructureFunctions: StructureFunctions as SF
using Base.Threads: Threads

println("Threads available: $(Threads.nthreads())")

N = Threads.nthreads() >= 4 ? 200_000 : 50_000
x = randn(N, 2)
u = randn(N, 2)
bins = collect(10.0:10.0:1000.0)
operator = SF.FullVectorStructureFunctionType{Float64}(order = 2)

serial_time = @elapsed begin
    SF.calculate_structure_function(
        operator,
        x,
        u,
        bins;
        backend = SF.SerialBackend(),
        show_progress = false,
        verbose = false,
    )
end

if Threads.nthreads() > 1
    threaded_result = nothing
    threaded_time = @elapsed begin
        threaded_result = SF.calculate_structure_function(
            operator,
            x,
            u,
            bins;
            backend = SF.ThreadedBackend(),
            show_progress = false,
            verbose = false,
        )
    end

    println("Serial time:   $(round(serial_time; digits = 3)) s")
    println("Threaded time: $(round(threaded_time; digits = 3)) s")
    println("Speedup:       $(round(serial_time / threaded_time; digits = 2))x")
    println("Pairs counted: $(sum(threaded_result.counts))")
else
    println("Run with JULIA_NUM_THREADS>1 to use SF.ThreadedBackend().")
end
