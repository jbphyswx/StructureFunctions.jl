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
        backend = SFC.SerialBackend(),
    )
    for (db, sums_ref, cnts_ref) in (
        (linear_dist, sums_lin_ref, cnts_lin_ref),
        begin
            sr = zeros(FT, 6, length(log_dist) - 1, n_val)
            cr = zeros(UInt32, 6, length(log_dist) - 1, n_val)
            SFC.calculate_structure_functions_single_pass_2d!(
                sr, cr, x, u, log_dist, value_bins_ntuple;
                backend = SFC.SerialBackend(),
            )
            (log_dist, sr, cr)
        end,
    )
        sums_gpu = zeros(FT, size(sums_ref)...)
        cnts_gpu = zeros(UInt32, size(cnts_ref)...)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_gpu, cnts_gpu, x, u, db, value_bins_ntuple;
            backend = SFC.GPUBackend(backend),
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
        backend = SFC.SerialBackend(),
    )
    sums_gpu = zeros(FT, 6, n_log, n_val_inf)
    cnts_gpu = zeros(UInt32, 6, n_log, n_val_inf)
    ws_inf = SFC.GPUSFWorkspace(backend, log_dist, inf_val; kind = :single_pass_2d)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, x, u, log_dist, inf_val;
        backend = SFC.GPUBackend(backend), workspace = ws_inf,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
    Test.@test cnts_gpu == cnts_ref

    ws2 = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    sums_ws = zeros(FT, 6, NB, n_val)
    cnts_ws = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ws, cnts_ws, x, u, linear_dist, value_bins_ntuple;
        backend = SFC.GPUBackend(backend), workspace = ws2,
    )
    Test.@test sums_ws ≈ sums_lin_ref atol = 1e-11
    Test.@test cnts_ws == cnts_lin_ref
    Test.@test ws2.partition_sums_dev === nothing
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
    GPUExt = Base.get_extension(SF, :StructureFunctionsGPUExt)
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
        backend = SFC.SerialBackend(),
    )
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    Test.@test ws.sp2d_accumulation_strategy.accum_mode == :typeplane
    Test.@test !ws.sp2d_accumulation_strategy.needs_partition_merge
    Test.@test ws.partition_sums_dev === nothing
    sums_gpu = zeros(FT, 6, NB, n_val)
    cnts_gpu = zeros(UInt32, 6, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, cnts_gpu, backend, x, u, linear_dist, value_bins_ntuple;
        workspace = ws,
    )
    Test.@test sums_gpu ≈ sums_ref atol = 1e-11
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
        backend = SFC.SerialBackend(),
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
    Test.@test ws.partition_sums_dev !== nothing
end
