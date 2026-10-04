using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH
using Random: Random

Random.seed!(42)

# A workspace gives the fresh-buffer answer, takes new same-shape inputs, and serves a varying-position joint batch.
Test.@testset "GPU Workspace & Slice Batch (KA.CPU)" begin
    N = 40
    T = 3
    FT = Float64
    x2 = rand(FT, 2, N)
    u2 = rand(FT, 2, N)
    x3 = rand(FT, 3, N)
    u3 = rand(FT, 3, N)

    linear_bins = collect(FT, range(0.0, 1.5, length = 11))
    log_bins = exp.(range(log(FT(0.01)), log(FT(1.5)), length = 11))
    value_bins = collect(FT, range(0.0, 2.0, length = 9))

    sft = SFT.L2SFType()
    backend = KA.CPU()
    G2 = SFH.FlatGeometry{2}()

    ref = SFC.gpu_calculate_structure_function(sft, backend, x3, u3, log_bins, UInt32; geometry = SFH.FlatGeometry{3}())
    out_ws = SFC.gpu_calculate_structure_function(sft, backend, x3, u3, log_bins, UInt32;
                                                  geometry = SFH.FlatGeometry{3}(),
                                                  workspace = SFC.GPUSFWorkspace(backend, log_bins))
    Test.@test ref.counts == out_ws.counts
    Test.@test ref.sums ≈ out_ws.sums atol = 1e-12

    Test.@testset "workspace input cache refreshes same-shape host inputs" begin
        ws_refresh = SFC.GPUSFWorkspace(backend, linear_bins)
        x_alt = reverse(x2; dims = 2)
        u_alt = 2 .* u2 .+ FT(0.25)

        ref_a = SFC.gpu_calculate_structure_function(
            sft, backend, x2, u2, linear_bins, UInt32; geometry = G2,
        )
        ref_b = SFC.gpu_calculate_structure_function(
            sft, backend, x_alt, u_alt, linear_bins, UInt32; geometry = G2,
        )
        out_a = SFC.gpu_calculate_structure_function(
            sft, backend, x2, u2, linear_bins, UInt32; geometry = G2, workspace = ws_refresh,
        )
        out_b = SFC.gpu_calculate_structure_function(
            sft, backend, x_alt, u_alt, linear_bins, UInt32; geometry = G2, workspace = ws_refresh,
        )

        Test.@test out_a.sums ≈ ref_a.sums atol = 1e-12
        Test.@test out_a.counts == ref_a.counts
        Test.@test out_b.sums ≈ ref_b.sums atol = 1e-12
        Test.@test out_b.counts == ref_b.counts
        Test.@test !isapprox(out_b.sums, ref_a.sums; atol = 1e-12)
        SFC.release!(ws_refresh)
    end

    x_batch = rand(FT, 2, N, T)
    u_batch = rand(FT, 2, N, T)
    n_dist = length(linear_bins) - 1
    n_val = length(value_bins) - 1
    sums_2d_ref = zeros(FT, n_dist, n_val, T)
    counts_2d_ref = zeros(UInt32, n_dist, n_val, T)
    for t in 1:T
        sf_t = SFC.gpu_calculate_structure_function_2d(
            sft, backend, x_batch[:, :, t], u_batch[:, :, t], linear_bins, value_bins, UInt32; geometry = G2,
        )
        sums_2d_ref[:, :, t] .= sf_t.sums
        counts_2d_ref[:, :, t] .= sf_t.counts
    end
    sums_2d_drv = zeros(FT, n_dist, n_val, T)
    counts_2d_drv = zeros(UInt32, n_dist, n_val, T)
    SFC.gpu_calculate_structure_function_2d_batch!(
        sums_2d_drv, counts_2d_drv, sft, backend, x_batch, u_batch, linear_bins, value_bins;
        geometry = G2, workspace = SFC.GPUSFWorkspace(backend, linear_bins, value_bins),
    )
    Test.@test sums_2d_drv ≈ sums_2d_ref atol = 1e-12
    Test.@test counts_2d_drv == counts_2d_ref
end

# Three-wide individual and single-pass batches equal the same slices one at a time, and count every pair once.
Test.@testset "three-dimensional slice batches (KA.CPU)" begin
    N, T, FT = 40, 3, Float64
    backend = KA.CPU()
    sft = SFT.L2SFType()
    bins = collect(FT, range(0.0, 1.8, length = 11))
    NB = length(bins) - 1
    Random.seed!(8801)
    x_batch = rand(FT, 3, N, T)
    u_batch = rand(FT, 3, N, T)

    ref_s = zeros(FT, NB, T)
    ref_c = zeros(UInt32, NB, T)
    for t in 1:T
        r = SFC.gpu_calculate_structure_function(sft, backend, x_batch[:, :, t], u_batch[:, :, t], bins, UInt32;
                                                 geometry = SFH.FlatGeometry{3}())
        ref_s[:, t] .= r.sums
        ref_c[:, t] .= r.counts
    end

    got_s = zeros(FT, NB, T)
    got_c = zeros(UInt32, NB, T)
    SFC.gpu_calculate_structure_function_batch!(got_s, got_c, sft, backend, x_batch, u_batch, bins;
                                                geometry = SFH.FlatGeometry{3}())
    Test.@test got_c == ref_c
    Test.@test isapprox(got_s, ref_s; atol = 1e-12)
    Test.@test sum(Int.(got_c)) == T * N * (N - 1) ÷ 2

    sp_s = zeros(FT, SFC.SINGLE_PASS_N, NB, T)
    sp_c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, T)
    SFC.gpu_calculate_structure_functions_single_pass_batch!(sp_s, sp_c, backend, x_batch, u_batch, bins;
                                                             geometry = SFH.FlatGeometry{3}())
    one_s = zeros(FT, SFC.SINGLE_PASS_N, NB, T)
    one_c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, T)
    for t in 1:T
        s, c = zeros(FT, SFC.SINGLE_PASS_N, NB), zeros(UInt32, SFC.SINGLE_PASS_N, NB)
        SFC.calculate_structure_functions_single_pass!(s, c, x_batch[:, :, t], u_batch[:, :, t], bins;
                                                       backend = CB.SerialBackend())
        one_s[:, :, t] .= s
        one_c[:, :, t] .= c
    end
    Test.@test all(t -> isapprox(sp_s[:, :, t], one_s[:, :, t]; rtol = 1e-10, atol = 1e-12), 1:T)
    Test.@test sp_c[1, :, :] == one_c[1, :, :]
end

const GPU_WS_FIXED_X_CASES = ((2, :typed), (3, :raw))

# A fixed-position slice batch matches serial at each width and bin spelling, and `!` adds into the caller's buffers.
Test.@testset "a fixed-position slice batch agrees at every width and bin spelling (KA.CPU)" begin
    N, T, FT = 60, 3, Float64
    backend = KA.CPU()
    sft = SFT.L2SFType()
    NB = 8
    Random.seed!(8802)
    bin_spellings = (typed = SF.LinearBinEdges(range(0.0, 1.0; length = NB + 1)),
                     raw = collect(range(0.0, 1.0; length = NB + 1)))
    Test.@testset "D = $D, $spelling bins" for (D, spelling) in GPU_WS_FIXED_X_CASES
        x = rand(FT, D, N)
        u = rand(FT, D, N, T)
        ref_s = zeros(FT, NB, T)
        ref_c = zeros(Int, NB, T)
        for t in 1:T
            r = SFC.calculate_structure_function(sft, x, u[:, :, t],
                collect(range(0.0, 1.0; length = NB + 1)), FT, SF.StructureFunctionSumsAndCounts;
                backend = CB.SerialBackend())
            ref_s[:, t] .= r.sums
            ref_c[:, t] .= r.counts
        end
        bins = bin_spellings[spelling]
        g = SFC.calculate_structure_function(sft, x, u, bins, SF.StructureFunctionSumsAndCounts;
            backend = CB.GPUBackend(backend))
        Test.@test reshape(collect(g.counts), NB, T) == ref_c
        Test.@test isapprox(reshape(collect(g.sums), NB, T), ref_s; rtol = 1e-10)
        s, c = zeros(FT, NB, T), zeros(UInt32, NB, T)
        for _ in 1:2
            SFC.calculate_structure_function_batch!(s, c, sft, x, u, bins; backend = CB.GPUBackend(backend))
        end
        Test.@test c == 2 .* ref_c
        Test.@test isapprox(s, 2 .* ref_s; rtol = 1e-10)
    end
end
