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

function _synthetic_value_bins_by_type(n_bins::Int)
    template = _synthetic_value_bins(n_bins)
    return [copy(template) for _ in 1:8]
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
    value_bins_by_type = _synthetic_value_bins_by_type(8)

    sft = SFT.L2SFType()
    backend = KA.CPU()
    NB = length(linear_bins) - 1

    # --- 1D: fresh alloc vs workspace ---
    for (x, u, bins) in ((x2, u2, linear_bins), (x2, u2, log_bins), (x3, u3, linear_bins))
        ref = SFC.gpu_calculate_structure_function(
            sft, backend, x, u, bins; return_sums_and_counts = true,
        )
        ws = SFC.GPUSFWorkspace(backend, bins)
        out_ws = SFC.gpu_calculate_structure_function(
            sft, backend, x, u, bins; return_sums_and_counts = true, workspace = ws,
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
            sft, backend, x, u, bins; return_sums_and_counts = true,
        )
        sums_k = zeros(eltype(ref.sums), NB)
        counts_k = zeros(UInt32, NB)
        for _ in 1:K
            SFC.gpu_calculate_structure_function!(sums_k, counts_k, sft, backend, x, u, bins)
        end
        Test.@test sums_acc ≈ sums_k atol = 1e-12
        Test.@test counts_acc ≈ counts_k
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
    ref_sums_sp, ref_counts_sp = SFC.calculate_structure_functions_single_pass(
        x2, u2, linear_bins; backend = SFC.GPUBackend(backend),
    )
    ws_sp = SFC.GPUSFWorkspace(backend, linear_bins; kind = :single_pass)
    out_sums_sp, out_counts_sp = SFC.calculate_structure_functions_single_pass(
        x2, u2, linear_bins; backend = SFC.GPUBackend(backend), workspace = ws_sp,
    )
    Test.@test ref_sums_sp[1:8, :] ≈ out_sums_sp[1:8, :] atol = 1e-12
    Test.@test ref_counts_sp[1:8, :] ≈ out_counts_sp[1:8, :]

    # --- single_pass_2d ---
    n_val = length(value_bins_by_type[1]) - 1
    sums_sp2d = zeros(FT, 8, NB, n_val)
    counts_sp2d = zeros(UInt32, 8, NB, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_sp2d, counts_sp2d, x2, u2, linear_bins, value_bins_by_type;
        backend = SFC.SerialBackend(),
    )
    ws_sp2d = SFC.GPUSFWorkspace(backend, linear_bins, value_bins_by_type)
    sums_gpu = zeros(FT, 8, NB, n_val)
    counts_gpu = zeros(UInt32, 8, NB, n_val)
    SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, backend, x2, u2, linear_bins, value_bins_by_type;
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
            sft, backend, x_batch[:, :, t], u_batch[:, :, t], linear_bins;
            return_sums_and_counts = true,
        )
        sums_slices[:, t] .= ref_t.sums
        counts_slices[:, t] .= ref_t.counts
    end

    sums_drv = zeros(FT, NB, T)
    counts_drv = zeros(UInt32, NB, T)
    ws_slice = SFC.GPUSFWorkspace(backend, linear_bins)
    SFC.gpu_calculate_structure_function_slices!(
        sums_drv, counts_drv, sft, backend, x_batch, u_batch, linear_bins;
        workspace = ws_slice,
    )
    Test.@test sums_drv ≈ sums_slices atol = 1e-12
    Test.@test counts_drv ≈ counts_slices

    SFC.calculate_structure_function_slices!(
        sums_drv, counts_drv, sft, x_batch, u_batch, linear_bins;
        backend = SFC.GPUBackend(backend), workspace = ws_slice,
    )
    Test.@test sums_drv ≈ sums_slices atol = 1e-12

    # --- flatten_grid_slices ---
    Ny, Nx = 5, 8
    x_grid = rand(FT, 2, Ny, Nx, T)
    u_grid = rand(FT, 2, Ny, Nx, T)
    x_flat, u_flat = SFC.flatten_grid_slices(x_grid, u_grid)
    Test.@test size(x_flat) == (2, Ny * Nx, T)
    Test.@test x_flat[:, 1, 1] ≈ x_grid[:, 1, 1, 1]
    Test.@test x_flat[:, 2, 1] ≈ x_grid[:, 2, 1, 1]

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
    SFC.gpu_calculate_structure_function_2d_slices!(
        sums_2d_drv, counts_2d_drv, sft, backend, x_batch, u_batch, linear_bins, value_bins;
        workspace = ws2d,
    )
    Test.@test sums_2d_drv ≈ sums_2d_ref atol = 1e-12
    Test.@test counts_2d_drv ≈ counts_2d_ref

    # --- single_pass slices ---
    sums_sp_ref = zeros(FT, 8, NB, T)
    counts_sp_ref = zeros(UInt32, 8, NB, T)
    for t in 1:T
        res_sums, res_counts = SFC.calculate_structure_functions_single_pass(
            x_batch[:, :, t], u_batch[:, :, t], linear_bins;
            backend = SFC.GPUBackend(backend),
        )
        sums_sp_ref[:, :, t] .= res_sums[1:8, :]
        counts_sp_ref[:, :, t] .= res_counts[1:8, :]
    end
    sums_sp_drv = zeros(FT, 8, NB, T)
    counts_sp_drv = zeros(UInt32, 8, NB, T)
    SFC.gpu_calculate_structure_functions_single_pass_slices!(
        sums_sp_drv, counts_sp_drv, backend, x_batch, u_batch, linear_bins;
        workspace = ws_sp,
    )
    Test.@test sums_sp_drv ≈ sums_sp_ref atol = 1e-10
    Test.@test counts_sp_drv ≈ counts_sp_ref

    # --- single_pass_2d slices ---
    sums_sp2d_ref = zeros(FT, 8, NB, n_val, T)
    counts_sp2d_ref = zeros(UInt32, 8, NB, n_val, T)
    for t in 1:T
        st = zeros(FT, 8, NB, n_val)
        ct = zeros(UInt32, 8, NB, n_val)
        SFC.calculate_structure_functions_single_pass_2d!(
            st, ct, x_batch[:, :, t], u_batch[:, :, t], linear_bins, value_bins_by_type;
            backend = SFC.SerialBackend(),
        )
        sums_sp2d_ref[:, :, :, t] .= st
        counts_sp2d_ref[:, :, :, t] .= ct
    end
    sums_sp2d_drv = zeros(FT, 8, NB, n_val, T)
    counts_sp2d_drv = zeros(UInt32, 8, NB, n_val, T)
    SFC.gpu_calculate_structure_functions_single_pass_2d_slices!(
        sums_sp2d_drv, counts_sp2d_drv, backend, x_batch, u_batch,
        linear_bins, value_bins_by_type; workspace = ws_sp2d,
    )
    Test.@test sums_sp2d_drv ≈ sums_sp2d_ref atol = 1e-10
    Test.@test counts_sp2d_drv ≈ counts_sp2d_ref

    SFC.release!(ws_slice)
    SFC.release!(ws2d)
    SFC.release!(ws_sp)
    SFC.release!(ws_sp2d)
end
