using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    InfPaddedBinEdges, LinearBinEdges, LogBinEdges, LogBinEdges_from_log_edges
using Random: Random

Random.seed!(2024)

function _synthetic_value_bins_ntuple(n_bins::Int, ::Type{FT} = Float64) where {FT}
    return ntuple(
        _ -> LinearBinEdges(range(FT(-1), FT(2); length = n_bins + 1)),
        6,
    )
end

const SP2D_EXT = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const SP2D_CAPS = SFC.gpu_device_caps(KA.CPU())

"""The strategy a call with `D`-wide points and sums of `FT`, counts of `CST`, takes, with the static
shared bytes of the kernel its mode compiles and of that kernel one cell wider than the mode's
widest."""
function _sp2d_strategy(n_dist::Int, n_val::Int, D::Int, ::Type{FT}, ::Type{CST} = UInt32) where {FT, CST}
    ext = SP2D_EXT
    cfg = ext._sp2d_accumulation_strategy(SP2D_CAPS, n_dist, n_val, D, FT, FT, CST)
    bytes_at(hc) = cfg.accum_mode === :typeplane ? ext._sp2d_typeplane_smem_bytes(FT, FT, CST, D, hc) :
                                                   ext._sp2d_sharedhist_smem_bytes(FT, FT, CST, D, hc)
    compiled = cfg.accum_mode === :direct ? ext._sp2d_direct_smem_bytes(FT, D) :
                                            bytes_at(ext._sp2d_sharedhist_compile_cells(cfg))
    return cfg, compiled, bytes_at(cfg.max_shared_cells), bytes_at(cfg.max_shared_cells + 1)
end

Test.@testset "GPU sp2d strategy fits the static budget it was chosen against" begin
    budget = SFC.gpu_static_smem_budget(SP2D_CAPS)
    for (nd, nv, D, FT, CST, mode) in (
        (10, 8, 2, Float64, UInt32, :shared),
        (50, 52, 2, Float64, UInt32, :typeplane),
        (50, 52, 2, Float32, UInt32, :typeplane),
        (30, 30, 2, Float64, UInt32, :typeplane),
        (30, 30, 2, Float64, Float64, :typeplane),
        (30, 30, 3, Float64, Float64, :typeplane),
        (60, 60, 2, Float64, UInt32, :direct),
        (50, 52, 2, Float64, Float64, :direct),
    )
        cfg, compiled, at_max, past_max = _sp2d_strategy(nd, nv, D, FT, CST)
        Test.@test cfg.accum_mode === mode
        Test.@test cfg.smem_budget == budget
        Test.@test cfg.n_joint_cells == SFC.SINGLE_PASS_N * nd * nv
        Test.@test compiled <= budget
        Test.@test at_max <= budget < past_max
        Test.@test cfg.needs_partition_merge == (mode === :direct)
        if mode === :typeplane
            Test.@test cfg.types_per_pass * cfg.plane_shared_cells <= cfg.max_shared_cells <
                       (cfg.types_per_pass + 1) * cfg.plane_shared_cells
            Test.@test cfg.n_type_passes == cld(SFC.SINGLE_PASS_N, cfg.types_per_pass)
        end
    end
end

Test.@testset "GPU sp2d HTP-EJ partitioned (KA.CPU)" begin
    backend = KA.CPU()
    N = 80
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    linear_dist = LinearBinEdges(range(FT(0.0), FT(1.5); length = 11))
    log_dist = LogBinEdges_from_log_edges(range(log(FT(0.01)), log(FT(1.5)); length = 11))
    value_bins_ntuple = _synthetic_value_bins_ntuple(8, FT)
    n_val = length(value_bins_ntuple[1]) - 1
    NB = length(linear_dist) - 1
    Test.@test first(_sp2d_strategy(NB, n_val, 2, FT)).accum_mode === :shared

    sums_lin_ref = zeros(FT, 6, NB, n_val)
    cnts_lin_ref = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_lin_ref, cnts_lin_ref, x, u, linear_dist, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    for (db, sums_ref, cnts_ref) in (
        (linear_dist, sums_lin_ref, cnts_lin_ref),
        begin
            sr = zeros(FT, 6, length(log_dist) - 1, n_val)
            cr = zeros(UInt32, 6, length(log_dist) - 1, n_val)
            SFC.calculate_structure_functions_single_pass_2d!(
                sr, cr, x, u, log_dist, value_bins_ntuple;
                backend = CB.SerialBackend(),
            )
            (log_dist, sr, cr)
        end,
    )
        sums_gpu = zeros(FT, size(sums_ref)...)
        cnts_gpu = zeros(UInt32, size(cnts_ref)...)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_gpu, cnts_gpu, x, u, db, value_bins_ntuple;
            backend = CB.GPUBackend(backend),
        )
        Test.@test sums_gpu ≈ sums_ref atol = 1e-11
        Test.@test cnts_gpu == cnts_ref

        sums_global = zeros(FT, size(sums_ref)...)
        cnts_global = zeros(UInt32, size(cnts_ref)...)
        SP2D_EXT._launch_single_pass_2d_kernel!(
            backend, 64, sums_global, cnts_global, x, u,
            SP2D_EXT._gpu_digitizer(backend, db, Val(:single_pass_2d)),
            SP2D_EXT._value_digitizer(nothing, backend, value_bins_ntuple),
            N, 2, length(db), SP2D_EXT._n_value_edges(value_bins_ntuple), SF.HelperFunctions.FlatGeometry{2}(),
        )
        KA.synchronize(backend)
        Test.@test sums_global ≈ sums_ref atol = 1e-11
        Test.@test cnts_global == cnts_ref
    end

    inner = LinearBinEdges(range(FT(-0.5), FT(1.5); length = n_val + 1))
    inf_val = InfPaddedBinEdges(inner)
    n_val_inf = length(inf_val) - 1
    n_log = length(log_dist) - 1
    sums_ref = zeros(FT, 6, n_log, n_val_inf)
    cnts_ref = zeros(UInt32, 6, n_log, n_val_inf)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, log_dist, inf_val;
        backend = CB.SerialBackend(),
    )
    sums_gpu = zeros(FT, 6, n_log, n_val_inf)
    cnts_gpu = zeros(UInt32, 6, n_log, n_val_inf)
    ws_inf = SFC.GPUSFWorkspace(backend, log_dist, inf_val; kind = :single_pass_2d)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, log_dist, inf_val;
        backend = CB.GPUBackend(backend), workspace = ws_inf,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
    Test.@test cnts_gpu == cnts_ref

    ws2 = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    sums_ws = zeros(FT, 6, NB, n_val)
    cnts_ws = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ws, cnts_ws, x, u, linear_dist, value_bins_ntuple;
        backend = CB.GPUBackend(backend), workspace = ws2,
    )
    Test.@test sums_ws ≈ sums_lin_ref atol = 1e-11
    Test.@test cnts_ws == cnts_lin_ref
    Test.@test ws2.lazy.partition_sums_dev === nothing
end

Test.@testset "GPU sp2d merge kernels (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    n_dist, n_val, n_blocks = 4, 3, 5
    partition_sums = rand(FT, 6, n_dist, n_val, n_blocks)
    ref_s = dropdims(sum(partition_sums; dims = 4); dims = 4)
    # Unweighted partitions count in UInt32 and the merge widens into the output's count type, so
    # block totals past typemax(UInt32) survive a UInt64 output; weighted partitions carry pair mass.
    for (PCT, OCT, draw) in ((UInt32, UInt32, () -> rand(UInt32(0):UInt32(1000))),
                             (UInt32, UInt64, () -> rand(UInt32(2)^31:typemax(UInt32))),
                             (Float64, Float64, () -> rand()))
        partition_counts = PCT[draw() for _ in 1:6, _ in 1:n_dist, _ in 1:n_val, _ in 1:n_blocks]
        ref_c = dropdims(sum(OCT.(partition_counts); dims = 4); dims = 4)
        for merge in (SP2D_EXT.SerialMerge(), SP2D_EXT.ParallelMerge())
            out_s = zeros(FT, 6, n_dist, n_val)
            out_c = zeros(OCT, 6, n_dist, n_val)
            SP2D_EXT._launch_merge_sp2d_partitions!(
                backend, out_s, out_c, partition_sums, partition_counts, n_dist, n_val, n_blocks, merge,
            )
            Test.@test out_s ≈ ref_s
            Test.@test out_c ≈ ref_c
            OCT <: Integer && Test.@test out_c == ref_c
        end
    end
end

Test.@testset "GPU sp2d typeplane mode (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    N = 64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    n_dist_bins = 30
    n_val_bins = 30
    linear_dist = LinearBinEdges(range(FT(0.0), FT(2.0); length = n_dist_bins + 1))
    value_bins_ntuple = _synthetic_value_bins_ntuple(n_val_bins, FT)
    NB = n_dist_bins
    n_val = n_val_bins
    Test.@test first(_sp2d_strategy(NB, n_val, 2, FT)).accum_mode === :typeplane

    sums_ref = zeros(FT, 6, NB, n_val)
    cnts_ref = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, linear_dist, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    sums_gpu = zeros(FT, 6, NB, n_val)
    cnts_gpu = zeros(UInt32, 6, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, backend, x, u, linear_dist, value_bins_ntuple;
        workspace = ws,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
    Test.@test cnts_gpu == cnts_ref
    Test.@test ws.lazy.partition_sums_dev === nothing
end

Test.@testset "GPU sp2d typeplane production shape log+infpadded (KA.CPU)" begin
    # Production LLC4320 SF shape: 50 log distance bins × 52 inf-padded linear value
    # bins in Float32. This is the first shape class that selects :typeplane AND the
    # log_linear/inflinear_cols kernel variant, whose flush helper takes lid::Int —
    # the combination that failed to compile on CUDA when @index returned Int32
    # (see the Int(@index(...)) bindings in ext/gpu). KA.CPU verifies numerics; the
    # Int32 dispatch itself is pinned by the script-hygiene test.
    backend = KA.CPU()
    FT = Float32
    N = 64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    n_dist_bins = 50
    log_dist = LogBinEdges_from_log_edges(
        range(log(FT(0.01)), log(FT(1.5)); length = n_dist_bins + 1)
    )
    inner = LinearBinEdges(range(FT(-0.5), FT(1.5); length = 51))
    inf_val = InfPaddedBinEdges(inner)
    n_val = length(inf_val) - 1
    Test.@test n_val == 52
    Test.@test first(_sp2d_strategy(n_dist_bins, n_val, 2, FT)).accum_mode === :typeplane

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

Test.@testset "GPU sp2d direct mode (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    N = 48
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    n_dist_bins = 60
    n_val_bins = 60
    linear_dist = LinearBinEdges(range(FT(0.0), FT(2.0); length = n_dist_bins + 1))
    value_bins_ntuple = _synthetic_value_bins_ntuple(n_val_bins, FT)
    NB = n_dist_bins
    n_val = n_val_bins
    Test.@test first(_sp2d_strategy(NB, n_val, 2, FT)).accum_mode === :direct

    sums_ref = zeros(FT, 6, NB, n_val)
    cnts_ref = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, linear_dist, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    sums_gpu = zeros(FT, 6, NB, n_val)
    cnts_gpu = zeros(UInt32, 6, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, backend, x, u, linear_dist, value_bins_ntuple;
        workspace = ws,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
    Test.@test cnts_gpu == cnts_ref
    Test.@test ws.lazy.partition_sums_dev !== nothing
end

# A weighted count is a pair mass, so the on-chip histogram and the block partitions hold it in the
# call's floating count type; every mode must then agree with the serial weighted histogram.
Test.@testset "GPU sp2d weighted, every mode (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    N = 48
    Random.seed!(20260925)
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    w = FT(0.25) .+ rand(FT, N)
    for (nd, nv, mode) in ((10, 8, :shared), (30, 30, :typeplane), (60, 60, :direct))
        Test.@test first(_sp2d_strategy(nd, nv, 2, FT, FT)).accum_mode === mode
        Test.@test first(_sp2d_strategy(nd, nv, 2, FT, UInt32)).accum_mode === mode
        dist = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vals = _synthetic_value_bins_ntuple(nv, FT)
        ref = SFC.calculate_structure_functions_single_pass_2d(
            x, u, dist, vals, FT; backend = CB.SerialBackend(), weights = w,
        )
        ws = SFC.GPUSFWorkspace(backend, dist, vals)
        for workspace in (nothing, ws, ws)
            got = SFC.calculate_structure_functions_single_pass_2d(
                x, u, dist, vals, FT;
                backend = CB.GPUBackend(backend), weights = w, workspace,
            )
            Test.@test keys(got) == keys(ref)
            for k in keys(ref)
                Test.@test collect(got[k].counts) ≈ collect(ref[k].counts) rtol = 1e-12
                Test.@test collect(got[k].sums) ≈ collect(ref[k].sums) rtol = 1e-12 atol = 1e-12
            end
        end
        # The same workspace serves an unweighted call afterwards: its partitions change count type.
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
        Test.@test cnts_gpu == cnts_ref
        Test.@test sums_gpu ≈ sums_ref rtol = 1e-12 atol = 1e-12
        mode === :direct && Test.@test eltype(ws.lazy.partition_counts_dev) === UInt32
    end
end

Test.@testset "GPU sp2d general distance edges take the tiled path (KA.CPU)" begin
    # Arbitrary (neither uniform nor log-uniform) distance edges digitize by device binary
    # search; the tiled shared-histogram path must agree with the CPU reference for every
    # combination of value-bin form and dimensionality.
    backend = KA.CPU()
    FT = Float32
    N, nd, nv = 96, 16, 8
    # r^1.7 on a uniform grid: strictly increasing, and neither spacing family matches it.
    dist_edges = collect(FT, range(FT(0), FT(1.5); length = nd + 1)) .^ FT(1.7)

    typed_val = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))
    raw_val = collect(FT, range(FT(-1), FT(2); length = nv + 1))
    inf_val = InfPaddedBinEdges(LinearBinEdges(range(FT(-0.5), FT(1.5); length = nv - 1)))

    Test.@testset "D = $D, $vname value bins" for D in (2, 3),
                                                  (vname, vb) in (
        ("typed", typed_val), ("raw", raw_val), ("infpadded", inf_val))
        Random.seed!(20260816 + D)
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        n_val = length(vb) - 1
        sums_ref = zeros(FT, 6, nd, n_val)
        cnts_ref = zeros(UInt32, 6, nd, n_val)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_ref, cnts_ref, x, u, dist_edges, vb; backend = CB.SerialBackend(),
        )

        ws = SFC.GPUSFWorkspace(backend, dist_edges, vb; kind = :single_pass_2d)
        sums_gpu = zeros(FT, 6, nd, n_val)
        cnts_gpu = zeros(UInt32, 6, nd, n_val)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_gpu, cnts_gpu, x, u, dist_edges, vb;
            backend = CB.GPUBackend(backend), workspace = ws,
        )
        Test.@test cnts_gpu == cnts_ref
        Test.@test sums_gpu ≈ sums_ref rtol = 1e-5 atol = 1e-6
    end
end

Test.@testset "GPU sp2d keeps data precision when bins are narrower (KA.CPU)" begin
    # The tiled kernel carries two element types: the coordinate tiles follow the data, the
    # shared histogram follows the output. Binding both from the bin scalars instead would
    # round Float64 coordinates to Float32 here, losing ~7 digits with nothing to show for it.
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
    # Float32 tiles would land near 1e-7 here; full Float64 throughout is orders tighter.
    Test.@test sums_gpu ≈ sums_ref rtol = 1e-12
end

Test.@testset "GPU batch entry points accept log distance bins (KA.CPU)" begin
    # The GPU batch entry points take the production LogBinEdges + InfPaddedBinEdges shape used by
    # varying-x conditioned batches. Varying-x (2, N, T) with per-slice coordinates.
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
    inner = LinearBinEdges(range(FT(-0.5), FT(1.5); length = 51))
    inf_val = InfPaddedBinEdges(inner)
    n_val = length(inf_val) - 1

    # sp2d batch (the production conditioned-run path)
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

    # sp1d batch with log bins
    sums1_ref = zeros(FT, 6, n_dist_bins, T)
    cnts1_ref = zeros(UInt32, 6, n_dist_bins, T)
    SFC.calculate_structure_functions_single_pass_batch!(
        sums1_ref, cnts1_ref, x, u, log_dist; backend = CB.SerialBackend(),
    )
    sums1_gpu = zeros(FT, 6, n_dist_bins, T)
    cnts1_gpu = zeros(UInt32, 6, n_dist_bins, T)
    SFC.calculate_structure_functions_single_pass_batch!(
        sums1_gpu, cnts1_gpu, x, u, log_dist; backend = CB.GPUBackend(backend),
    )
    Test.@test sums1_gpu ≈ sums1_ref rtol = 1e-5 atol = 1e-6
    Test.@test cnts1_gpu == cnts1_ref

    # individual 1D batch with log bins
    sf_type = SFT.LongitudinalSecondOrderStructureFunction
    sumsi_ref = zeros(FT, n_dist_bins, T)
    cntsi_ref = zeros(UInt32, n_dist_bins, T)
    SFC.calculate_structure_function_batch!(
        sumsi_ref, cntsi_ref, sf_type, x, u, log_dist; backend = CB.SerialBackend(),
    )
    sumsi_gpu = zeros(FT, n_dist_bins, T)
    cntsi_gpu = zeros(UInt32, n_dist_bins, T)
    SFC.calculate_structure_function_batch!(
        sumsi_gpu, cntsi_gpu, sf_type, x, u, log_dist; backend = CB.GPUBackend(backend),
    )
    Test.@test sumsi_gpu ≈ sumsi_ref rtol = 1e-5 atol = 1e-6
    Test.@test cntsi_gpu == cntsi_ref

    # fixed-x individual 1D with log bins → the fixed-x strip kernel
    x_fixed = x[:, :, 1]
    sumsf_ref = zeros(FT, n_dist_bins, T)
    cntsf_ref = zeros(UInt32, n_dist_bins, T)
    SFC.calculate_structure_function_batch!(
        sumsf_ref, cntsf_ref, sf_type, x_fixed, u, log_dist; backend = CB.SerialBackend(),
    )
    res = SFC.calculate_structure_function(sf_type, x_fixed, u, log_dist, SF.StructureFunctionSumsAndCounts;
        backend = CB.GPUBackend(backend))
    Test.@test res.sums ≈ sumsf_ref rtol = 1e-5 atol = 1e-6
    Test.@test res.counts == cntsf_ref
end

# Large 2D bin counts. Before these existed, nothing in the suite went past 60×60 on either axis,
# which is why two defects lived here unnoticed: SP2D threw outright for any `n_dist > 128` (the
# naive global-atomic route demanded a value-edge workspace that nothing supplies by default), and
# the `:direct` strategy was chosen over plain global atomics well past the point where it loses
# 2–4×. Float64 because Float32 carries only ~7 digits and a histogram this sparse (few pairs per
# cell, cancelling odd moments) disagrees with a Float64 reference by percent even on the CPU, so a
# Float32 assertion would be testing arithmetic and not the kernel.
Test.@testset "GPU sp2d large bin counts (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    N = 64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    for (nd, nv) in ((100, 100), (200, 200))
        dist = LinearBinEdges(range(FT(0), FT(1); length = nd + 1))
        val = _synthetic_value_bins_ntuple(nv, FT)
        sums_ref = zeros(FT, 6, nd, nv)
        cnts_ref = zeros(UInt32, 6, nd, nv)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_ref, cnts_ref, x, u, dist, val; backend = CB.SerialBackend(),
        )
        sums_gpu = zeros(FT, 6, nd, nv)
        cnts_gpu = zeros(UInt32, 6, nd, nv)
        # No workspace, which this path must accept.
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_gpu, cnts_gpu, x, u, dist, val; backend = CB.GPUBackend(backend),
        )
        Test.@test cnts_gpu == cnts_ref
        Test.@test sums_gpu ≈ sums_ref rtol = 1e-10 atol = 1e-12
    end
end

# The routing decision itself, not just the numbers a route produces. Every SP2D defect found so far
# was a *routing* fault that correctness assertions could not see, because each route computes the
# right answer — just at very different speeds.
Test.@testset "GPU sp2d strategy routing thresholds" begin
    ext = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    Test.@test ext !== nothing
    cells_bytes(nd, nv, ::Type{FT}) where {FT} = 6 * nd * nv * (sizeof(FT) + sizeof(UInt32))
    # Small histograms stay on chip; large ones must not select `:direct`, which loses to plain
    # global atomics above the measured crossover.
    Test.@test cells_bytes(60, 60, Float64) <= ext.SP2D_GLOBAL_ATOMIC_HIST_BYTES
    Test.@test cells_bytes(100, 100, Float64) > ext.SP2D_GLOBAL_ATOMIC_HIST_BYTES
    Test.@test cells_bytes(128, 128, Float32) > ext.SP2D_GLOBAL_ATOMIC_HIST_BYTES
    # 80×80 Float32 (307 KB) measured `:direct` still ahead by ~11%; it must stay below the cut.
    Test.@test cells_bytes(80, 80, Float32) <= ext.SP2D_GLOBAL_ATOMIC_HIST_BYTES
    for (nd, nv, FT, mode, global_atomics) in ((16, 8, Float32, :shared, false),
                                               (60, 60, Float64, :direct, false),
                                               (100, 100, Float64, :direct, true))
        cfg = first(_sp2d_strategy(nd, nv, 2, FT))
        Test.@test cfg.accum_mode === mode
        Test.@test ext._sp2d_prefers_global_atomics(cfg, FT, UInt32) === global_atomics
    end
end
