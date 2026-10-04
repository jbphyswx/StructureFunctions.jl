using Test: Test
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using Random: Random

Random.seed!(99)
const NSF_N, NSF_T, NSF_NB, NSF_NV = 8, 2, 6, 5
const NSF_PX, NSF_PU = randn(3, NSF_N), randn(3, NSF_N)
const NSF_X, NSF_U = randn(3, NSF_N, NSF_T), randn(3, NSF_N, NSF_T)
const NSF_BINS = collect(range(0.0, 4.0; length = NSF_NB + 1))
const NSF_VBINS = collect(range(-3.0, 3.0; length = NSF_NV + 1))
const NSF_OP = SFT.L2SFType()
const NSF_SCHEDULE = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
const NSF_GRID_U = randn(2, 64)
const NSF_GBINS = collect(range(0.0, 0.5; length = NSF_NB + 1))

"""Each extension's load flag, a backend it supplies, and the package its refusal names."""
const NSF_REFUSALS = (
    threads = (SFC._OHMYTHREADS_LOADED, CB.ThreadedBackend(), "OhMyThreads"),
    gpu = (SFC._KERNELABSTRACTIONS_LOADED, CB.GPUBackend(nothing), "KernelAbstractions"),
    distributed = (SFC._DISTRIBUTED_LOADED, CB.DistributedBackend(), "Distributed"),
    mpi = (SFC._MPI_LOADED, CB.MPIBackend(), "MPI"),
    distributed_threads = (SFC._OHMYTHREADS_LOADED, CB.DistributedBackend(CB.ThreadedBackend()), "OhMyThreads"),
)

# Public entries that take `backend`, each refused under one extension, every extension at least once.
const NSF_ENTRIES = (
    ("1-D", :gpu, be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("joint", :mpi, be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("1-D !", :gpu, be -> SFC.calculate_structure_function!(zeros(NSF_NB), zeros(Int, NSF_NB), NSF_OP, NSF_PX, NSF_PU,
                                                             NSF_BINS; backend = be)),
    ("joint !", :mpi, be -> SFC.calculate_structure_function!(zeros(NSF_NB, NSF_NV), zeros(Int, NSF_NB, NSF_NV), NSF_OP,
                                                               NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("single pass", :gpu, be -> SFC.calculate_structure_functions_single_pass(NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("single pass !", :mpi, be -> SFC.calculate_structure_functions_single_pass!(zeros(SFC.SINGLE_PASS_N, NSF_NB),
                                      zeros(Int, SFC.SINGLE_PASS_N, NSF_NB), NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("single pass 2D", :gpu, be -> SFC.calculate_structure_functions_single_pass_2d(NSF_PX, NSF_PU, NSF_BINS,
                                                                                    NSF_VBINS; backend = be)),
    ("single pass 2D !", :mpi, be -> SFC.calculate_structure_functions_single_pass_2d!(
                                         zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_NV),
                                         zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_NV),
                                         NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("tensor", :gpu, be -> SFC.calculate_structure_function_tensor(Val(2), NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("tensor !", :mpi, be -> SFC.calculate_structure_function_tensor!(zeros(3, 3, NSF_NB), zeros(Int, NSF_NB), Val(2),
                                                                       NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("multi-field", :gpu, be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, Fields(vectors = (NSF_PU,)),
                                                                 NSF_BINS; backend = be)),
    ("gridded lag sweep", :distributed_threads, be -> SFC.gridded_lag_sweep!(zeros(NSF_NB), zeros(Int, NSF_NB), NSF_OP,
                                                          NSF_GRID_U, NSF_SCHEDULE, NSF_GBINS, Val(2), Val(1), Val(0);
                                                          backend = be)),
    ("gridded lag sweep batch", :mpi, be -> SFC.gridded_lag_sweep_batch!(zeros(NSF_NB, 2), zeros(Int, NSF_NB, 2),
                                                NSF_OP, randn(2, 64, 2), NSF_SCHEDULE, NSF_GBINS, Val(2), Val(1),
                                                Val(0); backend = be)),
    ("slice batch", :threads, be -> SFC.calculate_structure_function_batch!(zeros(NSF_NB, NSF_T),
                                         zeros(Int, NSF_NB, NSF_T), NSF_OP, NSF_X, NSF_U, NSF_BINS; backend = be)),
    ("slice batch joint", :distributed, be -> SFC.calculate_structure_function_2d_batch!(
                                                 zeros(NSF_NB, NSF_NV, NSF_T), zeros(Int, NSF_NB, NSF_NV, NSF_T),
                                                 NSF_OP, NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend = be)),
    ("slice batch single pass", :threads, be -> SFC.calculate_structure_functions_single_pass_batch!(
                                                    zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_T),
                                                    zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_T),
                                                    NSF_X, NSF_U, NSF_BINS; backend = be)),
    ("slice batch single pass 2D", :threads, be -> SFC.calculate_structure_functions_single_pass_2d_batch!(
                                                       zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_NV, NSF_T),
                                                       zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_NV, NSF_T),
                                                       NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend = be)),
)

# An explicit backend raises an ArgumentError naming the package when the extension that supplies it is not loaded.
Test.@testset "an explicit backend is refused, naming its package, without the extension that supplies it" begin
    unrefused = Pair{String, Any}[]
    for (name, ext, run) in NSF_ENTRIES
        flag, backend, package = NSF_REFUSALS[ext]
        was = flag[]
        err = try
            flag[] = false
            run(backend)
            nothing
        catch e
            e
        finally
            flag[] = was
        end
        err isa ArgumentError && occursin("using $package", err.msg) || push!(unrefused, name => err)
    end
    Test.@test isempty(unrefused)
end

# Without the OhMyThreads extension, AutoBackend does not resolve to the threaded backend however many threads run.
Test.@testset "AutoBackend does not resolve to threads without the OhMyThreads extension" begin
    was = SFC._OHMYTHREADS_LOADED[]
    resolved = try
        SFC._OHMYTHREADS_LOADED[] = false
        SFC.resolve_auto_backend(; nthreads = 8)
    finally
        SFC._OHMYTHREADS_LOADED[] = was
    end
    Test.@test !(resolved isa CB.AbstractThreadedBackend)
end
