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
        8,
    )
end

"""Host reference for strip bucket policy (must match `SP2DPrivPolicy.jl`)."""
function _host_sp2d_priv_config(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = 8 * n_dist * n_val
    tile_overhead = 4 * 128 * sizeof(FT)
    reserve = 2048
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    buckets = (1024, 2048, 4096, 8192, 16384)
    smem_default = 48 * 1024
    smem_pref = 96 * 1024
    hist48 = max(0, smem_default - tile_overhead - reserve) ÷ cell_bytes
    bucket48 = first(b for b in buckets if b >= min(hist48, C))
    n48 = cld(C, bucket48)
    if n48 > 3
        hist = max(0, smem_pref - tile_overhead - reserve) ÷ cell_bytes
        bucket = first(b for b in buckets if b >= min(hist, C))
        return (C, bucket, cld(C, bucket), smem_pref)
    end
    return (C, bucket48, n48, smem_default)
end

Test.@testset "GPU sp2d HTP-EJ privatized (KA.CPU)" begin
    backend = KA.CPU()
    N = 80
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    linear_dist = LinearBinEdges(range(FT(0.0), FT(1.5); length = 11))
    # Log spacing: log-edge range → LogBinEdges (physical edges materialized once inside).
    log_dist = LogBinEdges_from_log_edges(range(log(FT(0.01)), log(FT(1.5)); length = 11))
    value_bins_ntuple = _synthetic_value_bins_ntuple(8, FT)
    n_val = length(value_bins_ntuple[1]) - 1
    NB = length(linear_dist) - 1

    # --- strip policy matches workspace metadata ---
    ws = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    cfg = ws.sp2d_priv_config
    C_ref, bucket_ref, nstrips_ref, smem_ref = _host_sp2d_priv_config(NB, n_val, FT)
    Test.@test cfg.n_joint_cells == C_ref
    Test.@test cfg.strip_bucket == bucket_ref
    Test.@test cfg.n_strips == nstrips_ref
    Test.@test cfg.smem_per_block == smem_ref

    # --- linear dist × 8-col linear value (production test grid) ---
    sums_lin_ref = zeros(FT, 8, NB, n_val)
    cnts_lin_ref = zeros(UInt32, 8, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_lin_ref, cnts_lin_ref, x, u, linear_dist, value_bins_ntuple;
        backend = SFC.SerialBackend(),
    )
    for (db, sums_ref, cnts_ref) in (
        (linear_dist, sums_lin_ref, cnts_lin_ref),
        begin
            sr = zeros(FT, 8, length(log_dist) - 1, n_val)
            cr = zeros(UInt32, 8, length(log_dist) - 1, n_val)
            SFC.calculate_structure_functions_single_pass_2d!(
                sr, cr, x, u, log_dist, value_bins_ntuple;
                backend = SFC.SerialBackend(),
            )
            (log_dist, sr, cr)
        end,
    )
        sums_priv = zeros(FT, size(sums_ref)...)
        cnts_priv = zeros(UInt32, size(cnts_ref)...)
        SFC.calculate_structure_functions_single_pass_2d!(
            sums_priv, cnts_priv, x, u, db, value_bins_ntuple;
            backend = SFC.GPUBackend(backend),
        )
        Test.@test sums_priv ≈ sums_ref atol = 1e-11
        Test.@test cnts_priv == cnts_ref

        sums_legacy = zeros(FT, size(sums_ref)...)
        cnts_legacy = zeros(UInt32, size(cnts_ref)...)
        gpu_be = SFC.GPUBackend(backend)
        SFC.gpu_calculate_structure_functions_single_pass_2d!(
            sums_legacy, cnts_legacy, backend, x, u, db, value_bins_ntuple;
            force_legacy = true,
        )
        Test.@test sums_legacy ≈ sums_ref atol = 1e-11
        Test.@test cnts_legacy == cnts_ref
    end

    # --- InfPadded shared value + log distance (SMODE-style; typed log dist for GPU) ---
    inner = LinearBinEdges(range(FT(-0.5), FT(1.5); length = n_val + 1))
    inf_val = InfPaddedBinEdges(inner)
    n_val_inf = length(inf_val) - 1
    n_log = length(log_dist) - 1
    sums_ref = zeros(FT, 8, n_log, n_val_inf)
    cnts_ref = zeros(UInt32, 8, n_log, n_val_inf)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, cnts_ref, x, u, log_dist, inf_val;
        backend = SFC.SerialBackend(),
    )
    sums_priv = zeros(FT, 8, n_log, n_val_inf)
    cnts_priv = zeros(UInt32, 8, n_log, n_val_inf)
    ws_inf = SFC.GPUSFWorkspace(backend, log_dist, inf_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_priv, cnts_priv, x, u, log_dist, inf_val;
        backend = SFC.GPUBackend(backend), workspace = ws_inf,
    )
    Test.@test sums_priv ≈ sums_ref atol = 1e-11
    Test.@test cnts_priv == cnts_ref

    # --- workspace reuse ---
    ws2 = SFC.GPUSFWorkspace(backend, linear_dist, value_bins_ntuple)
    sums_ws = zeros(FT, 8, NB, n_val)
    cnts_ws = zeros(UInt32, 8, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ws, cnts_ws, x, u, linear_dist, value_bins_ntuple;
        backend = SFC.GPUBackend(backend), workspace = ws2,
    )
    Test.@test sums_ws ≈ sums_lin_ref atol = 1e-11
    Test.@test cnts_ws == cnts_lin_ref
    Test.@test ws2.priv_sums_dev !== nothing
    Test.@test ws2.priv_n_tile_blocks > 0
end
