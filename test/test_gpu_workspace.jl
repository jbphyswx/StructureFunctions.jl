using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

Random.seed!(42)

"""Wide synthetic value-bin edges for unit tests only."""
function _synthetic_value_bins(n_bins::Int)
    return collect(range(-1.0, 2.0, length = n_bins + 1))
end

function _synthetic_value_bins_ntuple(n_bins::Int)
    template = _synthetic_value_bins(n_bins)
    return ntuple(_ -> copy(template), 6)
end

function _nan_equal(a, b; atol)
    axes(a) == axes(b) || return false
    return all(eachindex(a, b)) do i
        ai = a[i]
        bi = b[i]
        (isnan(ai) && isnan(bi)) || isapprox(ai, bi; atol = atol)
    end
end

Test.@testset "GPU Workspace & Slice Batch (KA.CPU)" begin
    N = 40
    T = 4
    FT = Float64
    x2 = rand(FT, 2, N)
    u2 = rand(FT, 2, N)
    x3 = rand(FT, 3, N)
    u3 = rand(FT, 3, N)

    linear_bins = collect(FT, range(0.0, 1.5, length = 11))
    log_bins = exp.(range(log(FT(0.01)), log(FT(1.5)), length = 11))
    value_bins = collect(FT, range(0.0, 2.0, length = 9))
    value_bins_ntuple = _synthetic_value_bins_ntuple(8)

    sft = SFT.L2SFType()
    backend = KA.CPU()
    NB = length(linear_bins) - 1

    # --- 1D: fresh alloc vs workspace ---
    for (x, u, bins) in ((x2, u2, linear_bins), (x2, u2, log_bins), (x3, u3, linear_bins))
        ref = SFC.gpu_calculate_structure_function(
            sft, backend, x, u, bins,
        )
        ws = SFC.GPUSFWorkspace(backend, bins)
        out_ws = SFC.gpu_calculate_structure_function(
            sft, backend, x, u, bins; workspace = ws,
        )
        Test.@test ref.counts ≈ out_ws.counts
        Test.@test ref.sums ≈ out_ws.sums atol = 1e-12

        sums_acc = zeros(eltype(ref.sums), NB)
        counts_acc = zeros(UInt32, NB)
        K = 3
        for _ in 1:K
            SFC.gpu_calculate_structure_function!(
                sums_acc, counts_acc, sft, backend, x, u, bins; workspace = ws,
            )
        end
        ref_k = SFC.gpu_calculate_structure_function(
            sft, backend, x, u, bins,
        )
        sums_k = zeros(eltype(ref.sums), NB)
        counts_k = zeros(UInt32, NB)
        for _ in 1:K
            SFC.gpu_calculate_structure_function!(sums_k, counts_k, sft, backend, x, u, bins)
        end
        Test.@test sums_acc ≈ sums_k atol = 1e-12
        Test.@test counts_acc ≈ counts_k
    end

    Test.@testset "workspace input cache refreshes same-shape host inputs" begin
        ws_refresh = SFC.GPUSFWorkspace(backend, linear_bins)
        x_alt = reverse(x2; dims = 2)
        u_alt = 2 .* u2 .+ FT(0.25)

        ref_a = SFC.gpu_calculate_structure_function(
            sft, backend, x2, u2, linear_bins,
        )
        ref_b = SFC.gpu_calculate_structure_function(
            sft, backend, x_alt, u_alt, linear_bins,
        )
        out_a = SFC.gpu_calculate_structure_function(
            sft, backend, x2, u2, linear_bins; workspace = ws_refresh,
        )
        out_b = SFC.gpu_calculate_structure_function(
            sft, backend, x_alt, u_alt, linear_bins; workspace = ws_refresh,
        )

        Test.@test out_a.sums ≈ ref_a.sums atol = 1e-12
        Test.@test out_a.counts ≈ ref_a.counts
        Test.@test out_b.sums ≈ ref_b.sums atol = 1e-12
        Test.@test out_b.counts ≈ ref_b.counts
        Test.@test !isapprox(out_b.sums, ref_a.sums; atol = 1e-12)
        SFC.release!(ws_refresh)
    end

    # --- 2D joint ---
    ref2d = SFC.gpu_calculate_structure_function_2d(
        sft, backend, x2, u2, linear_bins, value_bins,
    )
    ws2d = SFC.GPUSFWorkspace(backend, linear_bins, value_bins)
    out2d_ws = SFC.gpu_calculate_structure_function_2d(
        sft, backend, x2, u2, linear_bins, value_bins; workspace = ws2d,
    )
    Test.@test ref2d.sums ≈ out2d_ws.sums atol = 1e-12
    Test.@test ref2d.counts ≈ out2d_ws.counts

    # --- single_pass ---
    sp_inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
    ref_sp = SFC.calculate_structure_functions_single_pass(
        x2, u2, linear_bins; backend = CB.GPUBackend(backend),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    ws_sp = SFC.GPUSFWorkspace(backend, linear_bins; kind = :single_pass)
    out_sp = SFC.calculate_structure_functions_single_pass(
        x2, u2, linear_bins; backend = CB.GPUBackend(backend), workspace = ws_sp,
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    for k in sp_inv
        Test.@test _nan_equal(ref_sp[k].sums, out_sp[k].sums; atol = 1e-12)
        Test.@test ref_sp[k].counts ≈ out_sp[k].counts
    end

    # --- single_pass_2d ---
    n_val = length(value_bins_ntuple[1]) - 1
    sums_sp2d = zeros(FT, 6, NB, n_val)
    counts_sp2d = zeros(UInt32, 6, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_sp2d, counts_sp2d, x2, u2, linear_bins, value_bins_ntuple;
        backend = CB.SerialBackend(),
    )
    ws_sp2d = SFC.GPUSFWorkspace(backend, linear_bins, value_bins_ntuple)
    sums_gpu = zeros(FT, 6, NB, n_val)
    counts_gpu = zeros(UInt32, 6, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, backend, x2, u2, linear_bins, value_bins_ntuple;
        workspace = ws_sp2d,
    )
    Test.@test sums_gpu ≈ sums_sp2d atol = 1e-10
    Test.@test counts_gpu ≈ counts_sp2d

    # --- slice batch: (N_dims, N_points, T) ---
    x_batch = rand(FT, 2, N, T)
    u_batch = rand(FT, 2, N, T)
    sums_slices = zeros(FT, NB, T)
    counts_slices = zeros(UInt32, NB, T)

    for t in 1:T
        ref_t = SFC.gpu_calculate_structure_function(
            sft, backend, x_batch[:, :, t], u_batch[:, :, t], linear_bins,
        )
        sums_slices[:, t] .= ref_t.sums
        counts_slices[:, t] .= ref_t.counts
    end

    sums_drv = zeros(FT, NB, T)
    counts_drv = zeros(UInt32, NB, T)
    ws_slice = SFC.GPUSFWorkspace(backend, linear_bins)
    SFC.gpu_calculate_structure_function_batch!(
        sums_drv, counts_drv, sft, backend, x_batch, u_batch, linear_bins;
        workspace = ws_slice,
    )
    Test.@test sums_drv ≈ sums_slices atol = 1e-12
    Test.@test counts_drv ≈ counts_slices

    SFC.calculate_structure_function_batch!(
        sums_drv, counts_drv, sft, x_batch, u_batch, linear_bins;
        backend = CB.GPUBackend(backend), workspace = ws_slice,
    )
    Test.@test sums_drv ≈ sums_slices atol = 1e-12

    # --- 2D joint slices ---
    n_dist = length(linear_bins) - 1
    n_val = length(value_bins) - 1
    sums_2d_ref = zeros(FT, n_dist, n_val, T)
    counts_2d_ref = zeros(UInt32, n_dist, n_val, T)
    for t in 1:T
        sf_t = SFC.gpu_calculate_structure_function_2d(
            sft, backend, x_batch[:, :, t], u_batch[:, :, t], linear_bins, value_bins,
        )
        sums_2d_ref[:, :, t] .= sf_t.sums
        counts_2d_ref[:, :, t] .= sf_t.counts
    end
    sums_2d_drv = zeros(FT, n_dist, n_val, T)
    counts_2d_drv = zeros(UInt32, n_dist, n_val, T)
    SFC.gpu_calculate_structure_function_2d_batch!(
        sums_2d_drv, counts_2d_drv, sft, backend, x_batch, u_batch, linear_bins, value_bins;
        workspace = ws2d,
    )
    Test.@test sums_2d_drv ≈ sums_2d_ref atol = 1e-12
    Test.@test counts_2d_drv ≈ counts_2d_ref

    # --- single_pass slices ---
    sums_sp_ref = zeros(FT, 6, NB, T)
    counts_sp_ref = zeros(UInt32, 6, NB, T)
    for t in 1:T
        res = SFC.calculate_structure_functions_single_pass(
            x_batch[:, :, t], u_batch[:, :, t], linear_bins;
            backend = CB.GPUBackend(backend),
            output_type = SF.StructureFunctionSumsAndCounts,
        )
        for (i, k) in enumerate(sp_inv)
            sums_sp_ref[i, :, t] .= res[k].sums
            counts_sp_ref[i, :, t] .= res[k].counts
        end
    end
    sums_sp_drv = zeros(FT, 6, NB, T)
    counts_sp_drv = zeros(UInt32, 6, NB, T)
    SFC.gpu_calculate_structure_functions_single_pass_batch!(
        sums_sp_drv, counts_sp_drv, backend, x_batch, u_batch, linear_bins;
        workspace = ws_sp,
    )
    Test.@test sums_sp_drv ≈ sums_sp_ref atol = 1e-10
    Test.@test counts_sp_drv ≈ counts_sp_ref

    # --- single_pass_2d slices ---
    sums_sp2d_ref = zeros(FT, 6, NB, n_val, T)
    counts_sp2d_ref = zeros(UInt32, 6, NB, n_val, T)
    for t in 1:T
        st = zeros(FT, 6, NB, n_val)
        ct = zeros(UInt32, 6, NB, n_val)
        SFC.calculate_structure_functions_single_pass_2d!(
            st, ct, x_batch[:, :, t], u_batch[:, :, t], linear_bins, value_bins_ntuple;
            backend = CB.SerialBackend(),
        )
        sums_sp2d_ref[:, :, :, t] .= st
        counts_sp2d_ref[:, :, :, t] .= ct
    end
    sums_sp2d_drv = zeros(FT, 6, NB, n_val, T)
    counts_sp2d_drv = zeros(UInt32, 6, NB, n_val, T)
    SFC.gpu_calculate_structure_functions_single_pass_2d_batch!(
        sums_sp2d_drv, counts_sp2d_drv, backend, x_batch, u_batch,
        linear_bins, value_bins_ntuple; workspace = ws_sp2d,
    )
    Test.@test sums_sp2d_drv ≈ sums_sp2d_ref atol = 1e-10
    Test.@test counts_sp2d_drv ≈ counts_sp2d_ref

    SFC.release!(ws_slice)
    SFC.release!(ws2d)
    SFC.release!(ws_sp)
    SFC.release!(ws_sp2d)
end
