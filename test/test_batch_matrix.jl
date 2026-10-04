using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH,
    LinearBinEdges,
    batch_histograms_equal, pair_from_linear
using StructureFunctions.Calculations:
    auxiliary_shared_positions!, auxiliary_varying_positions!, serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!

Random.seed!(2025)

const SF_TYPE = SFT.L2SFType()
const FLAT2 = SFH.FlatGeometry{2}()
const CPU_BE = CB.SerialBackend()
const GPU_BE = CB.GPUBackend(KA.CPU())
const SP_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

function _rand_batch_fixed(N::Int, B::Int)
    FT = Float32
    x = rand(FT, 2, N)
    u = rand(FT, 2, N, B)
    edges = LinearBinEdges(0.0f0, 1.5f0, 11)
    return x, u, edges
end

function _rand_batch_varying(N::Int, B::Int)
    FT = Float32
    x = rand(FT, 2, N, B)
    u = rand(FT, 2, N, B)
    edges = LinearBinEdges(0.0f0, 1.8f0, 11)
    return x, u, edges
end

# pair_from_linear enumerates the upper triangle row by row, each pair i < j once.
Test.@testset "pair_from_linear" begin
    Test.@test all(N -> [pair_from_linear(k, N) for k in 1:(N * (N - 1) ÷ 2)] ==
                        [(i, j) for i in 1:(N - 1) for j in (i + 1):N], (2, 3, 50))
end

# Batch device paths on the KA.CPU backend reproduce the serial reference histograms.
Test.@testset "batch matrix parity (KA.CPU)" begin
    N, B = 24, 3

    # The shared-position 1D histogram matches serial for 17 slices, more than one strip of at most 16 fields holds.
    Test.@testset "1D fixed-x with more slices than one field strip" begin
        Bb = 17
        x, u, lbe = _rand_batch_fixed(N, Bb)
        NB = length(lbe) - 1
        cpu_s = zeros(Float32, NB, Bb)
        cpu_c = zeros(UInt32, NB, Bb)
        auxiliary_shared_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe; geometry = FLAT2)
        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe, SF.StructureFunctionSumsAndCounts;
            backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    # T3 and L2T1 batch histograms match serial for shared and varying positions and are not all zero.
    Test.@testset "signed transverse operators on the batch kernels" begin
        for sft in (SFT.T3SFType(), SFT.L2T1SFType())
            x, u, lbe = _rand_batch_fixed(N, B)
            NB = length(lbe) - 1
            cpu_s = zeros(Float32, NB, B)
            cpu_c = zeros(UInt32, NB, B)
            auxiliary_shared_positions!(cpu_s, cpu_c, x, u, sft, lbe; geometry = FLAT2)
            gpu_out = SFC.calculate_structure_function(
                sft, x, u, lbe, SF.StructureFunctionSumsAndCounts;
                backend = GPU_BE,
            )
            Test.@test any(!iszero, cpu_s) &&
                       batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)

            xv, uv, lbev = _rand_batch_varying(N, B)
            NBv = length(lbev) - 1
            cpu_sv = zeros(Float32, NBv, B)
            cpu_cv = zeros(UInt32, NBv, B)
            auxiliary_varying_positions!(cpu_sv, cpu_cv, xv, uv, sft, lbev; geometry = FLAT2)
            gpu_sv = zeros(Float32, NBv, B)
            gpu_cv = zeros(UInt32, NBv, B)
            SFC.calculate_structure_function_batch!(gpu_sv, gpu_cv, sft, xv, uv, lbev; backend = GPU_BE)
            Test.@test any(!iszero, cpu_sv) && batch_histograms_equal(gpu_sv, gpu_cv, cpu_sv, cpu_cv; atol = 1f-4)
        end
    end

    # The single-pass batch over shared positions equals the single pass of each slice for every invariant.
    Test.@testset "SP1D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        n_bins = length(lbe) - 1
        ref_s = zeros(Float32, 6, n_bins, B)
        ref_c = zeros(UInt32, 6, n_bins, B)
        for b in 1:B
            SFC.calculate_structure_functions_single_pass!(
                @view(ref_s[:, :, b]), @view(ref_c[:, :, b]),
                x, u[:, :, b], collect(lbe); backend = CPU_BE,
            )
        end
        gpu_sp = SFC.calculate_structure_functions_single_pass(
            x, u, lbe, SF.StructureFunctionSumsAndCounts; backend = GPU_BE,
        )
        Test.@test all(batch_histograms_equal(gpu_sp[k].sums, gpu_sp[k].counts, ref_s[t, :, :], ref_c[t, :, :];
                                              atol = 1f-4) for (t, k) in enumerate(SP_INV))
    end

    # The single-pass batch with per-slice positions matches the serial reference.
    Test.@testset "SP1D varying-x slices" begin
        x, u, lbe = _rand_batch_varying(N, B)
        n_bins = length(lbe) - 1
        cpu_s = zeros(Float32, 6, n_bins, B)
        cpu_c = zeros(UInt32, 6, n_bins, B)
        serial_calculate_structure_functions_single_pass!(cpu_s, cpu_c, x, u, lbe; geometry = FLAT2)

        gpu_s = zeros(Float32, 6, n_bins, B)
        gpu_c = zeros(UInt32, 6, n_bins, B)
        SFC.calculate_structure_functions_single_pass_batch!(
            gpu_s, gpu_c, x, u, lbe; backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    # The single-pass joint (distance x value) histogram with shared positions matches serial for every invariant.
    Test.@testset "SP2D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        val_edges = LinearBinEdges(-1.0f0, 1.0f0, 9)
        n_bins = length(lbe) - 1
        n_val = length(val_edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, B)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges; geometry = FLAT2)

        gpu_sp = SFC.calculate_structure_functions_single_pass_2d(
            x, u, lbe, val_edges; backend = GPU_BE,
        )
        Test.@test all(batch_histograms_equal(gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :, :], cpu_c[t, :, :, :];
                                              atol = 1f-4) for (t, k) in enumerate(SP_INV))
    end
end

# A CPU batch is the single-slice entry run once per slice in every family, over a covering of the batch axes.
const CPU_BATCH_CASES = (
    (Float64, 2, true, 3, false, SFC.NoCulling(), true, CB.SerialBackend()),
    (Float32, 3, true, SFC.BL_SLICE_LANES_MIN, true, SFC.AlwaysCulling(), false, CB.ThreadedBackend()),
    (Float64, 2, false, 3, false, SFC.NoCulling(), true, CB.SerialBackend()),
    (Float64, 2, true, 1, false, SFC.NoCulling(), true, CB.SerialBackend()),
)

Test.@testset "a CPU batch equals one call per slice in every family" begin
    sft = SFT.L2SFType()
    for (FT, D, shared, B, weighted, culling, linear, backend) in CPU_BATCH_CASES
        Random.seed!(hash((FT, D, shared, B, weighted)))
        N = 200
        db = collect(FT, range(0, 0.4; length = 9))
        vb = linear ? LinearBinEdges(FT(-0.2), FT(0.6), 7) : collect(FT, range(-0.2, 0.6; length = 7))
        vbs6 = ntuple(k -> linear ? LinearBinEdges(FT(-0.3 * k), FT(0.6), 7) :
                              collect(FT, range(-0.3 * k, 0.6; length = 7)), 6)
        nd, nv = length(db) - 1, length(vb) - 1
        x = shared ? rand(FT, D, N) : rand(FT, D, N, B)
        u = rand(FT, D, N, B)
        w = weighted ? rand(FT, N) .+ FT(0.5) : nothing
        CT = weighted ? FT : UInt32
        xs(t) = shared ? x : x[:, :, t]
        tol = FT == Float32 ? 1e-4 : 1e-10
        same(a, b) = isapprox(a, b; rtol = tol, atol = tol * maximum(abs, b; init = 0.0))
        kw = (; backend, culling, weights = w)
        one = (; backend = CB.SerialBackend(), weights = w)
        got, ref = (zeros(FT, nd, B), zeros(CT, nd, B)), (zeros(FT, nd, B), zeros(CT, nd, B))
        SFC.calculate_structure_function_batch!(got..., sft, x, u, db; kw...)
        for t in 1:B
            SFC.calculate_structure_function!(view(ref[1], :, t), view(ref[2], :, t), sft, xs(t), view(u, :, :, t), db;
                                              one...)
        end
        Test.@test same(got[1], ref[1]) && same(got[2], ref[2])
        for axis in (SFC.InvariantValueAxis(), SFC.SeparationAngleAxis(ones(FT, D)))
            ab = axis isa SFC.InvariantValueAxis ? vb : collect(FT, range(-0.01, π / 2 + 0.01; length = 7))
            got, ref = (zeros(FT, nd, nv, B), zeros(CT, nd, nv, B)), (zeros(FT, nd, nv, B), zeros(CT, nd, nv, B))
            SFC.calculate_structure_function_2d_batch!(got..., sft, x, u, db, ab; second_axis = axis, kw...)
            for t in 1:B
                SFC.calculate_structure_function!(view(ref[1], :, :, t), view(ref[2], :, :, t), sft, xs(t),
                                                  view(u, :, :, t), db, ab; second_axis = axis, one...)
            end
            Test.@test same(got[1], ref[1]) && same(got[2], ref[2])
        end
        got, ref = (zeros(FT, 6, nd, B), zeros(CT, 6, nd, B)), (zeros(FT, 6, nd, B), zeros(CT, 6, nd, B))
        SFC.calculate_structure_functions_single_pass_batch!(got..., x, u, db; kw...)
        for t in 1:B
            SFC.calculate_structure_functions_single_pass!(view(ref[1], :, :, t), view(ref[2], :, :, t), xs(t),
                                                           view(u, :, :, t), db; one...)
        end
        Test.@test same(got[1], ref[1]) && same(got[2], ref[2])
        for vbs in (vb, vbs6)
            got, ref = (zeros(FT, 6, nd, nv, B), zeros(CT, 6, nd, nv, B)), (zeros(FT, 6, nd, nv, B), zeros(CT, 6, nd, nv, B))
            SFC.calculate_structure_functions_single_pass_2d_batch!(got..., x, u, db, vbs; kw...)
            for t in 1:B
                SFC.calculate_structure_functions_single_pass_2d!(view(ref[1], :, :, :, t), view(ref[2], :, :, :, t),
                                                                  xs(t), view(u, :, :, t), db, vbs; one...)
            end
            Test.@test same(got[1], ref[1]) && same(got[2], ref[2])
        end
    end
end

# Every batch family refuses counts past the count type's range, from a nonzero start and from zero, writing nothing.
Test.@testset "batch counts past the count type are refused before anything is written" begin
    for N in (4, 24)
        rng = Random.MersenneTwister(932)
        u = randn(rng, 2, N, 2)
        x = rand(rng, 2, N)
        bins = SF.BinEdges([0.0, 2.0])
        value_bins = SF.BinEdges([-1e6, 1e6])
        initial = N == 4 ? UInt8(250) : UInt8(0)
        be = CB.SerialBackend()
        for (dims, call!) in (
            ((1, 2), (s, c) -> SFC.calculate_structure_function_batch!(s, c, SFT.S2SFType(), x, u, bins; backend = be)),
            ((1, 1, 2), (s, c) -> SFC.calculate_structure_function_2d_batch!(s, c, SFT.S2SFType(), x, u, bins,
                                                                             value_bins; backend = be)),
            ((6, 1, 2), (s, c) -> SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = be)),
            ((6, 1, 1, 2), (s, c) -> SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, bins,
                                                                                             value_bins; backend = be)))
            sums, counts = zeros(dims), fill(initial, dims)
            Test.@test_throws ArgumentError call!(sums, counts)
            Test.@test all(iszero, sums) && all(==(initial), counts)
        end
    end
end
