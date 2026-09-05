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

"""Host reference for SP2D accumulation strategy."""
function _host_sp2d_accumulation_strategy(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = 6 * n_dist * n_val
    plane = n_dist * n_val
    tile_overhead = 4 * 256 * sizeof(FT)
    meta = 5 * sizeof(Int)
    reserve = 2048
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    smem_default = 48 * 1024
    budget = smem_default - tile_overhead - meta - reserve
    max_shared = budget ÷ cell_bytes
    mode = if C <= max_shared
        :shared
    elseif plane <= max_shared
        :typeplane
    else
        :direct
    end
    tpp = mode == :typeplane ? min(6, max(1, max_shared ÷ plane)) : 6
    ntp = mode == :typeplane ? (6 + tpp - 1) ÷ tpp : 1
    needs_merge = mode == :direct
    return (C, mode, smem_default, max_shared, plane, tpp, ntp, needs_merge)
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

    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    cfg = ws.sp2d_accumulation_strategy
    C_ref, mode_ref, smem_ref, max_shared_ref, plane_ref, tpp_ref, ntp_ref, merge_ref = _host_sp2d_accumulation_strategy(NB, n_val, FT)
    Test.@test cfg.n_joint_cells == C_ref
    Test.@test cfg.accum_mode == mode_ref
    Test.@test cfg.smem_per_block == smem_ref
    Test.@test cfg.max_shared_cells == max_shared_ref
    Test.@test cfg.plane_cells == plane_ref
    Test.@test cfg.types_per_pass == tpp_ref
    Test.@test cfg.n_type_passes == ntp_ref
    Test.@test cfg.needs_partition_merge == merge_ref
    Test.@test !cfg.needs_partition_merge
    Test.@test mode_ref == :shared

    C50, mode50, smem50, max_shared50, plane50, tpp50, ntp50 = _host_sp2d_accumulation_strategy(50, 52, FT)
    Test.@test mode50 == :typeplane
    Test.@test C50 == SFC.SINGLE_PASS_N * 50 * 52
    Test.@test plane50 == 50 * 52
    Test.@test C50 > max_shared50
    Test.@test plane50 <= max_shared50
    Test.@test smem50 == 48 * 1024
    _, mode50f, _, _, _, tpp50f, ntp50f = _host_sp2d_accumulation_strategy(50, 52, Float32)
    Test.@test mode50f == :typeplane
    Test.@test tpp50f == 2
    Test.@test ntp50f == cld(SFC.SINGLE_PASS_N, tpp50f)

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
        SFC.gpu_calculate_structure_functions_single_pass_2d!(
            sums_global, cnts_global, backend, x, u, db, value_bins_ntuple;
            force_global_atomic = true,
        )
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
    Test.@test ws2.sp2d_pair_kernel !== nothing
    Test.@test !ws2.sp2d_accumulation_strategy.needs_partition_merge
end

Test.@testset "GPU sp2d merge kernels (KA.CPU)" begin
    backend = KA.CPU()
    FT = Float64
    n_dist, n_val, n_blocks = 4, 3, 5
    C = 6 * n_dist * n_val
    partition_sums = rand(FT, 6, n_dist, n_val, n_blocks)
    partition_counts = rand(UInt32, 6, n_dist, n_val, n_blocks)
    out_s = zeros(FT, 6, n_dist, n_val)
    out_c = zeros(UInt32, 6, n_dist, n_val)
    ref_s = zeros(FT, 6, n_dist, n_val)
    ref_c = zeros(UInt32, 6, n_dist, n_val)
    for t in 1:6, d in 1:n_dist, v in 1:n_val, b in 1:n_blocks
        ref_s[t, d, v] += partition_sums[t, d, v, b]
        ref_c[t, d, v] += partition_counts[t, d, v, b]
    end
    GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    for mode in (:serial, :parallel)
        fill!(out_s, 0)
        fill!(out_c, 0)
        GPUExt._launch_merge_sp2d_partitions!(
            backend, out_s, out_c, partition_sums, partition_counts, n_dist, n_val, n_blocks;
            merge_mode = mode,
        )
        Test.@test out_s ≈ ref_s
        Test.@test out_c == ref_c
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
    _, mode_ref, _, _, _, _, _ = _host_sp2d_accumulation_strategy(NB, n_val, FT)
    Test.@test mode_ref == :typeplane

    sums_ref = zeros(FT, 6, NB, n_val)
    cnts_ref = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, linear_dist, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    Test.@test ws.sp2d_accumulation_strategy.accum_mode == :typeplane
    Test.@test !ws.sp2d_accumulation_strategy.needs_partition_merge
    Test.@test ws.lazy.partition_sums_dev === nothing
    sums_gpu = zeros(FT, 6, NB, n_val)
    cnts_gpu = zeros(UInt32, 6, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, backend, x, u, linear_dist, value_bins_ntuple;
        workspace = ws,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
    Test.@test cnts_gpu == cnts_ref
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
    _, mode_ref, _, _, _, _, _ = _host_sp2d_accumulation_strategy(n_dist_bins, n_val, FT)
    Test.@test mode_ref == :typeplane

    sums_ref = zeros(FT, 6, n_dist_bins, n_val)
    cnts_ref = zeros(UInt32, 6, n_dist_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, log_dist, inf_val;
        backend = CB.SerialBackend(),
    )

    ws = SFC.GPUSFWorkspace(backend, log_dist, inf_val; kind = :single_pass_2d)
    Test.@test ws.sp2d_accumulation_strategy.accum_mode == :typeplane
    Test.@test !ws.sp2d_accumulation_strategy.needs_partition_merge
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
    _, mode_ref, _, _, _, _, _, merge_ref = _host_sp2d_accumulation_strategy(NB, n_val, FT)
    Test.@test mode_ref == :direct
    Test.@test merge_ref

    sums_ref = zeros(FT, 6, NB, n_val)
    cnts_ref = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, linear_dist, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    Test.@test ws.sp2d_accumulation_strategy.accum_mode == :direct
    Test.@test ws.sp2d_accumulation_strategy.needs_partition_merge
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

Test.@testset "GPU sp2d general distance edges take the tiled path (KA.CPU)" begin
    # Arbitrary (neither uniform nor log-uniform) distance edges digitize by device binary
    # search. Before the :general route existed these fell through to the global-atomic
    # kernel; the tiled shared-histogram path must agree with the CPU reference for every
    # combination of value-bin form and dimensionality it now covers.
    backend = KA.CPU()
    FT = Float32
    N, nd, nv = 96, 16, 8
    # r^1.7 on a uniform grid: strictly increasing, and neither spacing family matches it.
    dist_edges = collect(FT, range(FT(0), FT(1.5); length = nd + 1)) .^ FT(1.7)
    GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    Test.@test GPUExt._sp2d_dist_variant(dist_edges) == :general

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
        # A resolved pair kernel is what distinguishes the tiled route from the fallback.
        Test.@test ws.sp2d_pair_kernel !== nothing
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
    # The GPU batch entry points route through the unified device path, whose
    # _sf_batch_dist_digitizer handles linear, log, and general edges. Guard
    # against regressing to the old linear-only host check (which rejected the
    # production LogBinEdges + InfPaddedBinEdges shape used by varying-x
    # conditioned batches). Varying-x (2, N, T) with per-slice coordinates.
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

    # Typed FMA fast-path routing: linear AND log qualify (same 5-param FMA
    # digitize, log in log space); raw vectors take the exact general digitizer.
    GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    Test.@test GPUExt._fma_distance_bins(log_dist) === log_dist
    lin = LinearBinEdges(range(FT(0.0), FT(1.5); length = n_dist_bins + 1))
    Test.@test GPUExt._fma_distance_bins(lin) === lin
    Test.@test GPUExt._fma_distance_bins(collect(range(0.0, 1.5; length = 11))) === nothing

    # fixed-x individual 1D with log bins → the tiled FMA fast kernel (Val{LOG}=true)
    x_fixed = x[:, :, 1]
    sumsf_ref = zeros(FT, n_dist_bins, T)
    cntsf_ref = zeros(UInt32, n_dist_bins, T)
    SFC.calculate_structure_function_batch!(
        sumsf_ref, cntsf_ref, sf_type, x_fixed, u, log_dist; backend = CB.SerialBackend(),
    )
    res = SFC.gpu_calculate_structure_function_batch(sf_type, backend, x_fixed, u, log_dist)
    Test.@test res.sums ≈ sumsf_ref rtol = 1e-5 atol = 1e-6
    Test.@test res.counts == cntsf_ref
end

# Large 2D bin counts. Before these existed, nothing in the suite went past 60×60 on either axis,
# which is why two defects lived here unnoticed: SP2D threw outright for any `n_dist > 128` (the
# naive global-atomic route demanded a value-edge workspace that nothing supplies by default), and
# the `:direct` strategy was chosen over plain global atomics well past the point where it loses
# 2–4× (see `gpu/SPEED_OF_LIGHT.md`). Float64 because Float32 carries only ~7 digits and a histogram
# this sparse (few pairs per cell, cancelling odd moments) disagrees with a Float64 reference by
# percent even on the CPU — a Float32 assertion here would be testing arithmetic, not the kernel.
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
        # No workspace: the path that used to raise `ArgumentError` here.
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
    for (nd, nv, FT) in ((16, 8, Float32), (60, 60, Float64), (100, 100, Float64))
        cfg = ext._sp2d_accumulation_strategy(nd, nv, FT, SFC.gpu_device_caps(nothing))
        Test.@test cfg.accum_mode in (:shared, :typeplane, :direct)
        Test.@test cfg.n_joint_cells == 6 * nd * nv
    end
end
