# Parity matrix for production batch fast paths (KA.CPU).
using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH,
    LinearBinEdges,
    batch_histograms_equal, batch_max_abs_diff, pair_from_linear
using StructureFunctions.Calculations:
    auxiliary_shared_positions!, auxiliary_varying_positions!, serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!, auxiliary_joint2d!

Random.seed!(2025)

const SF_TYPE = SFT.L2SFType()
const FLAT2 = SFH.FlatGeometry{2}()
const CPU_BE = CB.SerialBackend()
const GPU_BE = CB.GPUBackend(KA.CPU())

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

# Every batch fast path on the KA.CPU backend reproduces the serial reference histograms.
Test.@testset "batch matrix parity (KA.CPU)" begin
    N, B = 24, 3

    # pair_from_linear returns an ordered pair of valid indices at the ends and middle of a 20000-point pair list.
    Test.@testset "pair_from_linear large N" begin
        Nbig = 20_000
        total = Nbig * (Nbig - 1) ÷ 2
        for k in (1, 2, total ÷ 2, total - 1, total)
            i, j = pair_from_linear(k, Nbig)
            Test.@test 1 <= i < j <= Nbig
        end
    end

    # The 1D histogram of a batch sharing one position set matches the serial shared-position reference.
    Test.@testset "row1 individual 1D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        NB = length(lbe) - 1
        cpu_s = zeros(Float32, NB, B)
        cpu_c = zeros(UInt32, NB, B)
        auxiliary_shared_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe; geometry = FLAT2)

        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe, SF.StructureFunctionSumsAndCounts;
            backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    # The shared-position 1D histogram matches serial for a batch of 17 slices.
    Test.@testset "row1b individual 1D fixed-x with a 17-slice batch" begin
        Nb, Bb = 24, 17
        x, u, lbe = _rand_batch_fixed(Nb, Bb)
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

    # The 1D batch histogram with per-slice positions matches the serial varying-position reference.
    Test.@testset "row2 individual 1D varying-x" begin
        x, u, lbe = _rand_batch_varying(N, B)
        NB = length(lbe) - 1
        cpu_s = zeros(Float32, NB, B)
        cpu_c = zeros(UInt32, NB, B)
        auxiliary_varying_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe; geometry = FLAT2)

        gpu_s = zeros(Float32, NB, B)
        gpu_c = zeros(UInt32, NB, B)
        SFC.calculate_structure_function_batch!(
            gpu_s, gpu_c, SF_TYPE, x, u, lbe; backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    # T3 and L2T1 batch histograms match serial for shared and varying positions and are not all zero.
    Test.@testset "row2b signed transverse operators on the batch kernels" begin
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
            Test.@test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
            Test.@test any(!iszero, cpu_s)

            xv, uv, lbev = _rand_batch_varying(N, B)
            NBv = length(lbev) - 1
            cpu_sv = zeros(Float32, NBv, B)
            cpu_cv = zeros(UInt32, NBv, B)
            auxiliary_varying_positions!(cpu_sv, cpu_cv, xv, uv, sft, lbev; geometry = FLAT2)
            gpu_sv = zeros(Float32, NBv, B)
            gpu_cv = zeros(UInt32, NBv, B)
            SFC.calculate_structure_function_batch!(gpu_sv, gpu_cv, sft, xv, uv, lbev; backend = GPU_BE)
            Test.@test batch_histograms_equal(gpu_sv, gpu_cv, cpu_sv, cpu_cv; atol = 1f-4)
            Test.@test any(!iszero, cpu_sv)
        end
    end

    # Serial batch single-pass equals per-slice single-pass, and the backend entry matches both for every invariant.
    Test.@testset "row3 SP1D fixed-x" begin
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
        cpu_s = zeros(Float32, 6, n_bins, B)
        cpu_c = zeros(UInt32, 6, n_bins, B)
        serial_calculate_structure_functions_single_pass!(cpu_s, cpu_c, x, u, lbe; geometry = FLAT2)
        Test.@test batch_histograms_equal(cpu_s, cpu_c, ref_s, ref_c)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass(
            x, u, lbe, SF.StructureFunctionSumsAndCounts; backend = GPU_BE,
        )
        for (t, k) in enumerate(inv)
            Test.@test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :], cpu_c[t, :, :]; atol = 1f-4,
            )
        end
    end

    # The single-pass batch with per-slice positions matches the serial reference.
    Test.@testset "row4 SP1D varying-x slices" begin
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
    Test.@testset "row5 SP2D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        val_edges = LinearBinEdges(-1.0f0, 1.0f0, 9)
        n_bins = length(lbe) - 1
        n_val = length(val_edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, B)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges; geometry = FLAT2)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass_2d(
            x, u, lbe, val_edges; backend = GPU_BE,
        )
        for (t, k) in enumerate(inv)
            Test.@test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :, :], cpu_c[t, :, :, :]; atol = 1f-4,
            )
        end
    end

    # The single-pass joint histogram batch with per-slice positions matches the serial reference.
    Test.@testset "row6 SP2D varying-x slices" begin
        x, u, lbe = _rand_batch_varying(N, B)
        val_edges = LinearBinEdges(-1.0f0, 1.0f0, 9)
        n_bins = length(lbe) - 1
        n_val = length(val_edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, B)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges; geometry = FLAT2)

        gpu_s = zeros(Float32, 6, n_bins, n_val, B)
        gpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            gpu_s, gpu_c, x, u, lbe, val_edges; backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    # The L2 joint (distance x value) histogram of a shared-position batch matches serial.
    Test.@testset "row7 joint 2D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        val_edges = LinearBinEdges(-0.5f0, 1.5f0, 9)
        n_bins = length(lbe) - 1
        n_val = length(val_edges) - 1
        cpu_s = zeros(Float32, n_bins, n_val, B)
        cpu_c = zeros(UInt32, n_bins, n_val, B)
        auxiliary_joint2d!(cpu_s, cpu_c, SF_TYPE, x, u, lbe, val_edges; geometry = FLAT2)

        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe, val_edges; backend = GPU_BE,
        )
        Test.@test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    # The single-pass joint histogram on a 10 x 50 bin grid matches serial for every invariant.
    Test.@testset "row8 SP2D fixed-x 10 x 50 bin grid" begin
        Np, Bp = 32, 2
        x, u, lbe = _rand_batch_fixed(Np, Bp)
        val_edges = LinearBinEdges(-1.0f0, 1.0f0, 51)
        n_bins = length(lbe) - 1
        n_val = length(val_edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, Bp)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, Bp)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges; geometry = FLAT2)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass_2d(
            x, u, lbe, val_edges; backend = GPU_BE,
        )
        for (t, k) in enumerate(inv)
            Test.@test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :, :], cpu_c[t, :, :, :]; atol = 1f-4,
            )
        end
    end
end

# A CPU batch is the single-slice entry run once per slice: every family, shared positions on both sides of the
# slices-innermost threshold and positions varying per slice, weighted, culled or not, value edges a vector or linear
# (the value columns formed in the vectorized pass), serial and threaded.
Test.@testset "a CPU batch equals one call per slice in every family" begin
    sft = SFT.L2SFType()
    for FT in (Float64, Float32), D in (2, 3), shared in (true, false), B in (1, 3, SFC.BL_SLICE_LANES_MIN + 1),
        weighted in (false, true), culling in (SFC.NoCulling(), SFC.AlwaysCulling()), linear in (false, true),
        backend in (CB.SerialBackend(), CB.ThreadedBackend())
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
        xs(t) = shared ? x : view(x, :, :, t)
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
