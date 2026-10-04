using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, InfPaddedBinEdges, LinearBinEdges, LogBinEdges_from_log_edges
using Random: Random

Random.seed!(2024)

function _synthetic_value_bins_ntuple(n_bins::Int, ::Type{FT} = Float64) where {FT}
    return ntuple(
        _ -> LinearBinEdges(range(FT(-1), FT(2); length = n_bins + 1)),
        6,
    )
end

const SP2D_EXT = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)

"""Whether every invariant of `got` matches `ref`: counts to `rtol`, sums to `rtol` and `atol`."""
_sp_agrees(got, ref; rtol, atol = 0.0) = keys(got) == keys(ref) &&
    all(k -> isapprox(collect(got[k].counts), collect(ref[k].counts); rtol) &&
             isapprox(collect(got[k].sums), collect(ref[k].sums); rtol, atol), keys(ref))

# 50 log distance bins by 52 inf-padded linear value bins in Float32, a type-plane histogram, match serial.
Test.@testset "GPU sp2d type-plane histogram, log distance and inf-padded value bins (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float32
    N = 64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    n_dist_bins = 50
    log_dist = LogBinEdges_from_log_edges(
        range(log(FT(0.01)), log(FT(1.5)); length = n_dist_bins + 1)
    )
    inf_val = InfPaddedBinEdges(LinearBinEdges(range(FT(-0.5), FT(1.5); length = 51)))
    n_val = length(inf_val) - 1

    sums_ref = zeros(FT, 6, n_dist_bins, n_val)
    cnts_ref = zeros(UInt32, 6, n_dist_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, log_dist, inf_val;
        backend = CB.SerialBackend(),
    )

    ws = SFC.GPUSFWorkspace(backend, log_dist, inf_val; kind = :single_pass_2d)
    sums_gpu = zeros(FT, 6, n_dist_bins, n_val)
    cnts_gpu = zeros(UInt32, 6, n_dist_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, log_dist, inf_val;
        backend = CB.GPUBackend(backend), workspace = ws,
    )
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-5 atol = 1e-6
    Test.@test cnts_gpu == cnts_ref
end

# Per accumulation mode, weighted calls match serial fresh or on a reused workspace, which then serves unweighted.
Test.@testset "GPU sp2d weighted, every mode (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    N = 48
    Random.seed!(20260925)
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    w = FT(0.25) .+ rand(FT, N)
    for (nd, nv) in ((10, 8), (30, 30), (60, 60))
        dist = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vals = _synthetic_value_bins_ntuple(nv, FT)
        ref = SFC.calculate_structure_functions_single_pass_2d(
            x, u, dist, vals, FT; backend = CB.SerialBackend(), weights = w,
        )
        ws = SFC.GPUSFWorkspace(backend, dist, vals)
        got = [SFC.calculate_structure_functions_single_pass_2d(
                   x, u, dist, vals, FT; backend = CB.GPUBackend(backend), weights = w, workspace,
               ) for workspace in (nothing, ws, ws)]
        Test.@test (nd, nv, all(g -> _sp_agrees(g, ref; rtol = 1e-12, atol = 1e-12), got)) == (nd, nv, true)
        cnts_ref = zeros(UInt32, 6, nd, nv)
        sums_ref = zeros(FT, 6, nd, nv)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_ref, cnts_ref, x, u, dist, vals; backend = CB.SerialBackend(),
        )
        cnts_gpu = zeros(UInt32, 6, nd, nv)
        sums_gpu = zeros(FT, 6, nd, nv)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_gpu, cnts_gpu, x, u, dist, vals;
            backend = CB.GPUBackend(backend), workspace = ws,
        )
        Test.@test (nd, nv, cnts_gpu == cnts_ref, isapprox(sums_gpu, sums_ref; rtol = 1e-12, atol = 1e-12)) ==
                   (nd, nv, true, true)
    end
end

# Distance edges r^1.7, neither linear nor logarithmic, with inf-padded value bins in 3D match serial.
Test.@testset "GPU sp2d general distance edges (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float32
    D, N, nd, nv = 3, 96, 16, 8
    dist_edges = collect(FT, range(FT(0), FT(1.5); length = nd + 1)) .^ FT(1.7)
    vb = InfPaddedBinEdges(LinearBinEdges(range(FT(-0.5), FT(1.5); length = nv - 1)))
    Random.seed!(20260816 + D)
    x = rand(FT, D, N)
    u = rand(FT, D, N)
    sums_ref = zeros(FT, 6, nd, nv)
    cnts_ref = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, dist_edges, vb; backend = CB.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, dist_edges, vb; kind = :single_pass_2d)
    sums_gpu = zeros(FT, 6, nd, nv)
    cnts_gpu = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, dist_edges, vb;
        backend = CB.GPUBackend(backend), workspace = ws,
    )
    Test.@test cnts_gpu == cnts_ref
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-5 atol = 1e-6
end

# Float64 coordinates with Float32 bin edges reproduce the Float64 serial histogram to Float64 precision.
Test.@testset "GPU sp2d keeps data precision when bins are narrower (KA.CPU)" begin
    backend = KA.CPU()
    N, nd, nv = 64, 10, 8
    Random.seed!(20260816)
    x = rand(Float64, 2, N)
    u = rand(Float64, 2, N)
    db = LinearBinEdges(range(Float32(0), Float32(1.5); length = nd + 1))
    vb = _synthetic_value_bins_ntuple(nv, Float32)

    sums_ref = zeros(Float64, 6, nd, nv)
    cnts_ref = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, db, vb; backend = CB.SerialBackend(),
    )
    sums_gpu = zeros(Float64, 6, nd, nv)
    cnts_gpu = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, db, vb; backend = CB.GPUBackend(backend),
    )
    Test.@test cnts_gpu == cnts_ref
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-12
end

# The batch entry with log distance and inf-padded value bins, past the shared fit, matches serial per slice.
Test.@testset "GPU sp2d batch past the shared fit, log distance bins (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float32
    N = 40
    T = 3
    x = rand(FT, 2, N, T)
    u = rand(FT, 2, N, T)
    n_dist_bins = 50
    log_dist = LogBinEdges_from_log_edges(
        range(log(FT(0.01)), log(FT(1.5)); length = n_dist_bins + 1)
    )
    inf_val = InfPaddedBinEdges(LinearBinEdges(range(FT(-0.5), FT(1.5); length = 51)))
    n_val = length(inf_val) - 1

    sums_ref = zeros(FT, 6, n_dist_bins, n_val, T)
    cnts_ref = zeros(UInt32, 6, n_dist_bins, n_val, T)
    for t in 1:T
        SFC.calculate_structure_functions_single_pass_2d!(
            view(sums_ref, :, :, :, t), view(cnts_ref, :, :, :, t),
            x[:, :, t], u[:, :, t], log_dist, inf_val;
            backend = CB.SerialBackend(),
        )
    end
    sums_gpu = zeros(FT, 6, n_dist_bins, n_val, T)
    cnts_gpu = zeros(UInt32, 6, n_dist_bins, n_val, T)
    SFC.calculate_structure_functions_single_pass_2d_batch!(
        sums_gpu, cnts_gpu, x, u, log_dist, inf_val;
        backend = CB.GPUBackend(backend),
    )
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-5 atol = 1e-6
    Test.@test cnts_gpu == cnts_ref
end

# A distance axis past the tiled kernel's bin cap matches serial.
Test.@testset "GPU sp2d distance bins past the tiled cap (KA.CPU)" begin
    FT = Float64
    N = 64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    nd, nv = SP2D_EXT.SF_GPU_MAX_BINS + 1, 8
    dist = LinearBinEdges(range(FT(0), FT(1); length = nd + 1))
    val = _synthetic_value_bins_ntuple(nv, FT)
    sums_ref = zeros(FT, 6, nd, nv)
    cnts_ref = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, dist, val; backend = CB.SerialBackend(),
    )
    sums_gpu = zeros(FT, 6, nd, nv)
    cnts_gpu = zeros(UInt32, 6, nd, nv)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, dist, val; backend = CB.GPUBackend(KA.CPU()),
    )
    Test.@test cnts_gpu == cnts_ref
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-10 atol = 1e-12
end
