using Test: Test
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using Random: Random

# An explicit backend either does what it says or raises naming the package that supplies it; only
# `AutoBackend` chooses, and it only ever names a backend that can run the request. The extension
# load flag is a `Ref`, so the unloaded state is reachable here and restored afterwards.

Random.seed!(99)
const NSF_N, NSF_T, NSF_NB, NSF_NV = 40, 2, 6, 5
const NSF_X = randn(3, NSF_N, NSF_T)
const NSF_U = randn(3, NSF_N, NSF_T)
const NSF_BINS = collect(range(0.0, 4.0; length = NSF_NB + 1))
const NSF_VBINS = collect(range(-3.0, 3.0; length = NSF_NV + 1))
const NSF_OP = SFT.L2SFType()
const NSF_SCHEDULE = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
const NSF_GRID_U = reshape(randn(2, 8, 8), 2, :)
const NSF_GBINS = collect(range(0.0, 0.5; length = NSF_NB + 1))

# Every entry that reaches a driver supplied by the OhMyThreads extension. `run(backend)` returns
# the arrays that entry produces, so each row is checked against its own serial answer. A threaded
# entry added without a guard fails here as soon as it is listed.
const NSF_THREADED_ENTRIES = (
    (name = "calculate_structure_function_batch!",
     run = function (backend)
         s, c = zeros(NSF_NB, NSF_T), zeros(Int, NSF_NB, NSF_T)
         SFC.calculate_structure_function_batch!(s, c, NSF_OP, NSF_X, NSF_U, NSF_BINS; backend)
         Any[s, c]
     end),
    (name = "calculate_structure_function_2d_batch!",
     run = function (backend)
         s, c = zeros(NSF_NB, NSF_NV, NSF_T), zeros(Int, NSF_NB, NSF_NV, NSF_T)
         SFC.calculate_structure_function_2d_batch!(s, c, NSF_OP, NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend)
         Any[s, c]
     end),
    (name = "calculate_structure_functions_single_pass_batch!",
     run = function (backend)
         s, c = zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_T), zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_T)
         SFC.calculate_structure_functions_single_pass_batch!(s, c, NSF_X, NSF_U, NSF_BINS; backend)
         Any[s, c]
     end),
    (name = "calculate_structure_functions_single_pass_2d_batch!",
     run = function (backend)
         s = zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_NV, NSF_T)
         c = zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_NV, NSF_T)
         SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend)
         Any[s, c]
     end),
    (name = "calculate_structure_function! over auxiliary axes",
     run = function (backend)
         s, c = zeros(NSF_NB, NSF_T), zeros(Int, NSF_NB, NSF_T)
         SFC.calculate_structure_function!(s, c, NSF_OP, NSF_X, NSF_U, NSF_BINS; backend)
         Any[s, c]
     end),
    (name = "joint calculate_structure_function! over auxiliary axes",
     run = function (backend)
         s, c = zeros(NSF_NB, NSF_NV, NSF_T), zeros(Int, NSF_NB, NSF_NV, NSF_T)
         SFC.calculate_structure_function!(s, c, NSF_OP, NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend)
         Any[s, c]
     end),
    (name = "calculate_structure_functions_single_pass over auxiliary axes",
     run = function (backend)
         r = SFC.calculate_structure_functions_single_pass(NSF_X, NSF_U, NSF_BINS; backend)
         vcat(Any[r[k].sums for k in keys(r)], Any[r[k].counts for k in keys(r)])
     end),
    (name = "calculate_structure_functions_single_pass_2d over auxiliary axes",
     run = function (backend)
         r = SFC.calculate_structure_functions_single_pass_2d(NSF_X, NSF_U, NSF_BINS, NSF_VBINS; backend)
         vcat(Any[r[k].sums for k in keys(r)], Any[r[k].counts for k in keys(r)])
     end),
    (name = "gridded_lag_sweep!",
     run = function (backend)
         s, c = zeros(NSF_NB), zeros(Int, NSF_NB)
         SFC.gridded_lag_sweep!(s, c, NSF_OP, NSF_GRID_U, NSF_SCHEDULE, NSF_GBINS,
                                Val(2), Val(1), Val(0); backend)
         Any[s, c]
     end),
)

"Equal counts exactly, sums to round-off; an empty bin reads `NaN` on both sides or on neither."
function nsf_same(a, b)
    size(a) == size(b) || return false
    for (x, y) in zip(a, b)
        if isnan(x) || isnan(y)
            isnan(x) && isnan(y) || return false
        elseif abs(x - y) > 1e-12 * max(abs(y), 1e-12)
            return false
        end
    end
    return true
end

nsf_nontrivial(arrays) = any(a -> any(v -> isfinite(v) && !iszero(v), a), arrays)

Test.@testset "an explicit threaded backend is refused without the OhMyThreads extension" begin
    was = SFC._OHMYTHREADS_LOADED[]
    references = map(e -> e.run(CB.SerialBackend()), NSF_THREADED_ENTRIES)
    Test.@test all(nsf_nontrivial, references)

    try
        SFC._OHMYTHREADS_LOADED[] = false
        for (entry, ref) in zip(NSF_THREADED_ENTRIES, references)
            Test.@testset "$(entry.name)" begin
                err = try
                    entry.run(CB.ThreadedBackend())
                    nothing
                catch e
                    e
                end
                Test.@test err isa ArgumentError
                Test.@test err isa ArgumentError && occursin("OhMyThreads", err.msg)

                # `AutoBackend` chooses, so it must still produce the serial answer.
                auto = entry.run(CB.AutoBackend())
                Test.@test all(p -> nsf_same(p[1], p[2]), zip(auto, ref))
            end
        end

        # `Auto` never resolves to a threaded backend when the extension cannot supply one.
        Test.@test !(SFC.resolve_auto_backend(; nthreads = 8) isa CB.AbstractThreadedBackend)
        Test.@test !(SFC.resolve_auto_backend() isa CB.AbstractThreadedBackend)
    finally
        SFC._OHMYTHREADS_LOADED[] = was
    end

    # With the extension loaded the same explicit request runs and agrees with serial.
    Test.@test SFC._ohmythreads_loaded()
    for (entry, ref) in zip(NSF_THREADED_ENTRIES, references)
        thr = entry.run(CB.ThreadedBackend())
        Test.@test all(p -> nsf_same(p[1], p[2]), zip(thr, ref))
    end
end

const NSF_PX = randn(3, NSF_N)
const NSF_PU = randn(3, NSF_N)

# Every public entry that takes `backend`, reduced to the call; the backend is the only argument that
# varies, so a refusal can come from nothing but the boundary check.
const NSF_BACKEND_ENTRIES = (
    ("1-D", be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("joint", be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("1-D !", be -> SFC.calculate_structure_function!(zeros(NSF_NB), zeros(Int, NSF_NB), NSF_OP, NSF_PX, NSF_PU,
                                                       NSF_BINS; backend = be)),
    ("joint !", be -> SFC.calculate_structure_function!(zeros(NSF_NB, NSF_NV), zeros(Int, NSF_NB, NSF_NV), NSF_OP,
                                                         NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("single pass", be -> SFC.calculate_structure_functions_single_pass(NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("single pass !", be -> SFC.calculate_structure_functions_single_pass!(zeros(SFC.SINGLE_PASS_N, NSF_NB),
                                zeros(Int, SFC.SINGLE_PASS_N, NSF_NB), NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("single pass 2D", be -> SFC.calculate_structure_functions_single_pass_2d(NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS;
                                                                              backend = be)),
    ("single pass 2D !", be -> SFC.calculate_structure_functions_single_pass_2d!(
                                   zeros(SFC.SINGLE_PASS_N, NSF_NB, NSF_NV), zeros(Int, SFC.SINGLE_PASS_N, NSF_NB, NSF_NV),
                                   NSF_PX, NSF_PU, NSF_BINS, NSF_VBINS; backend = be)),
    ("tensor", be -> SFC.calculate_structure_function_tensor(Val(2), NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("tensor !", be -> SFC.calculate_structure_function_tensor!(zeros(3, 3, NSF_NB), zeros(Int, NSF_NB), Val(2),
                                                                 NSF_PX, NSF_PU, NSF_BINS; backend = be)),
    ("multi-field", be -> SFC.calculate_structure_function(NSF_OP, NSF_PX, Fields(vectors = (NSF_PU,)), NSF_BINS;
                                                           backend = be)),
    ("gridded lag sweep", be -> SFC.gridded_lag_sweep!(zeros(NSF_NB), zeros(Int, NSF_NB), NSF_OP, NSF_GRID_U,
                                    NSF_SCHEDULE, NSF_GBINS, Val(2), Val(1), Val(0); backend = be)),
    ("gridded lag sweep batch", be -> SFC.gridded_lag_sweep_batch!(zeros(NSF_NB, 2), zeros(Int, NSF_NB, 2), NSF_OP,
                                          randn(2, 64, 2), NSF_SCHEDULE, NSF_GBINS, Val(2), Val(1), Val(0); backend = be)),
    (map(e -> (e.name, be -> e.run(be)), NSF_THREADED_ENTRIES[1:4])...),
)

Test.@testset "an explicit backend is refused, naming its package, without the extension that supplies it" begin
    families = (
        (SFC._OHMYTHREADS_LOADED, CB.ThreadedBackend(), "OhMyThreads"),
        (SFC._KERNELABSTRACTIONS_LOADED, CB.GPUBackend(nothing), "KernelAbstractions"),
        (SFC._DISTRIBUTED_LOADED, CB.DistributedBackend(), "Distributed"),
        (SFC._MPI_LOADED, CB.MPIBackend(), "MPI"),
        # a wrapper is available only when its local backend is
        (SFC._OHMYTHREADS_LOADED, CB.DistributedBackend(CB.ThreadedBackend()), "OhMyThreads"),
    )
    for (flag, backend, package) in families, (name, run) in NSF_BACKEND_ENTRIES
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
        Test.@test (name, package, err isa ArgumentError && occursin("using $package", err.msg)) ==
                   (name, package, true)
    end
end

Test.@testset "with the extensions loaded, a call no method takes is a MethodError" begin
    Test.@test SFC._ohmythreads_loaded()
    Test.@test_throws MethodError SFC.threaded_calculate_structure_function!(NSF_OP)
    Test.@test_throws MethodError SFC.gpu_calculate_structure_function_batch!(NSF_OP)
    Test.@test_throws MethodError SFC.threaded_calculate_structure_function_tensor!(NSF_OP)
    Test.@test_throws MethodError SFC.threaded_calculate_structure_function(NSF_OP, NSF_PX, NSF_PU, NSF_BINS, UInt32, 1)
end

Test.@testset "core defines no threaded driver that runs serially" begin
    # A core method on one of these names would run on one task while claiming to thread, which is
    # what `_require_threading` and the extension's methods exist to prevent.
    for f in (SFC.auxiliary_structure_function_threaded!,
              SFC.auxiliary_joint2d_threaded!,
              SFC.threaded_calculate_structure_functions_single_pass!,
              SFC.threaded_calculate_structure_functions_single_pass_2d!)
        Test.@test !any(m -> m.module === SFC, methods(f))
    end
end

Test.@testset "the single-pass siblings agree on what they return" begin
    # Both entries default to the raw accumulator, so a caller who reads `.counts` off one reads it
    # off the other.
    r1 = SFC.calculate_structure_functions_single_pass(NSF_X, NSF_U, NSF_BINS)
    r2 = SFC.calculate_structure_functions_single_pass_2d(NSF_X, NSF_U, NSF_BINS, NSF_VBINS)
    Test.@test keys(r1) == keys(r2)
    for k in keys(r2)
        Test.@test hasproperty(r1[k], :sums) && hasproperty(r1[k], :counts)
        Test.@test hasproperty(r2[k], :sums) && hasproperty(r2[k], :counts)
    end
end
