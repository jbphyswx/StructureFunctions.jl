using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using Test: Test

Test.@testset "Tensor Structure Functions" begin
    x = [0.0 1.0 0.0; 0.0 0.0 1.0]
    u = [0.0 2.0 3.0; 0.0 5.0 7.0]
    bins = [0.0, 1.1, 2.0]

    t2 = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test t2 isa SF.StructureFunctionTensorSumsAndCounts{2}
    Test.@test size(t2.sums) == (2, 2, 2)
    Test.@test size(t2.counts) == (2,)
    Test.@test t2.counts == UInt32[2, 1]

    du12 = u[:, 2] - u[:, 1]
    du13 = u[:, 3] - u[:, 1]
    du23 = u[:, 3] - u[:, 2]
    expected_bin1 = du12 * du12' + du13 * du13'
    expected_bin2 = du23 * du23'
    Test.@test t2.sums[:, :, 1] ≈ expected_bin1
    Test.@test t2.sums[:, :, 2] ≈ expected_bin2

    # Default output is the averaged mean tensor D_ij(r) = sums ./ counts (per distance bin).
    t2_mean = SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = CB.SerialBackend())
    Test.@test t2_mean isa SF.StructureFunctionTensor{2}
    Test.@test t2_mean.values[:, :, 1] ≈ expected_bin1 ./ 2
    Test.@test t2_mean.values[:, :, 2] ≈ expected_bin2 ./ 1

    s2 = SFC.calculate_structure_function(
        SFT.S2SF, x, u, bins; backend = CB.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts
    )
    trace_sums = [sum(t2.sums[a, a, bin] for a in 1:2) for bin in axes(t2.sums, 3)]
    Test.@test trace_sums ≈ s2.sums
    Test.@test t2.counts == s2.counts

    t3 = SFC.calculate_structure_function_tensor(
        Val(3), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test t3 isa SF.StructureFunctionTensorSumsAndCounts{3}
    Test.@test size(t3.sums) == (2, 2, 2, 2)
    Test.@test t3.counts == t2.counts
    Test.@test t3.sums[1, 2, 2, 1] ≈ du12[1] * du12[2] * du12[2] +
        du13[1] * du13[2] * du13[2]

    u_aux = cat(u, 2u; dims = 3)
    t2_aux = SFC.calculate_structure_function_tensor(
        Val(2), x, u_aux, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test size(t2_aux.sums) == (2, 2, 2, 2)
    Test.@test size(t2_aux.counts) == (2, 2)
    Test.@test t2_aux.sums[:, :, :, 1] ≈ t2.sums
    Test.@test t2_aux.sums[:, :, :, 2] ≈ 4 .* t2.sums
end

Test.@testset "KHM Diagnostics" begin
    r = [1.0, 2.0, 3.0, 4.0]
    DLL = r .^ 2
    DTT = DLL .+ r .* SF.KHM.finite_difference(r, DLL) ./ 2
    Test.@test SF.KHM.transverse_incompressibility_residual(r, DLL, DTT; dimension = 3) ≈ zeros(4)

    ε = 0.2
    S3 = .-(4 / 5) .* ε .* r
    Test.@test SF.KHM.epsilon_from_four_fifths(r, S3) ≈ fill(ε, length(r))
    Test.@test SF.KHM.four_fifths_residual(r, S3, ε) ≈ zeros(length(r))
end

Test.@testset "each inertial-range law takes the quantity it is stated for" begin
    # §9.4: the four-fifths law is for ⟨δu_L³⟩ (L3SF) and the four-thirds law for ⟨δu_L‖δu‖²⟩
    # (S3SF). They differ by 5/3, so handing one function the other's quantity returns a wrong ε
    # that looks entirely plausible. These pin that the two are distinct and each inverts its own law.
    r = collect(range(0.1, 2.0; length = 9))
    eps = 0.37

    # a field obeying the four-fifths law exactly
    L3 = -(4 / 5) .* eps .* r
    Test.@test SF.KHM.epsilon_from_four_fifths(r, L3) ≈ fill(eps, length(r))
    Test.@test all(abs.(SF.KHM.four_fifths_residual(r, L3, eps)) .< 1e-12)

    # a field obeying the four-thirds law exactly
    S3 = -(4 / 3) .* eps .* r
    Test.@test SF.KHM.epsilon_from_four_thirds(r, S3) ≈ fill(eps, length(r))
    Test.@test all(abs.(SF.KHM.four_thirds_residual(r, S3, eps)) .< 1e-12)

    # the two are NOT interchangeable: feeding one quantity to the other's inverse is off by 5/3
    wrong = SF.KHM.epsilon_from_four_fifths(r, S3)
    Test.@test all(wrong ./ eps .≈ 5 / 3)

    # Yaglom returns the scalar dissipation, on its own law
    eps_th = 0.21
    LS2 = -(4 / 3) .* eps_th .* r
    Test.@test SF.KHM.epsilon_theta_from_yaglom(r, LS2) ≈ fill(eps_th, length(r))
    Test.@test all(abs.(SF.KHM.yaglom_residual(r, LS2, eps_th)) .< 1e-12)
end
