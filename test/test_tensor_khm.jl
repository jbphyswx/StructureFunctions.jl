using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using Test: Test

Test.@testset "Tensor Structure Functions" begin
    x = [0.0 1.0 0.0; 0.0 0.0 1.0]
    u = [0.0 2.0 3.0; 0.0 5.0 7.0]
    bins = [0.0, 1.1, 2.0]

    t2 = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins; backend = SFC.SerialBackend(),
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
    t2_mean = SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = SFC.SerialBackend())
    Test.@test t2_mean isa SF.StructureFunctionTensor{2}
    Test.@test t2_mean.values[:, :, 1] ≈ expected_bin1 ./ 2
    Test.@test t2_mean.values[:, :, 2] ≈ expected_bin2 ./ 1

    s2 = SFC.calculate_structure_function(
        SFT.S2SF, x, u, bins; backend = SFC.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts
    )
    trace_sums = [sum(t2.sums[a, a, bin] for a in 1:2) for bin in axes(t2.sums, 3)]
    Test.@test trace_sums ≈ s2.sums
    Test.@test t2.counts == s2.counts

    t3 = SFC.calculate_structure_function_tensor(
        Val(3), x, u, bins; backend = SFC.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test t3 isa SF.StructureFunctionTensorSumsAndCounts{3}
    Test.@test size(t3.sums) == (2, 2, 2, 2)
    Test.@test t3.counts == t2.counts
    Test.@test t3.sums[1, 2, 2, 1] ≈ du12[1] * du12[2] * du12[2] +
        du13[1] * du13[2] * du13[2]

    u_aux = cat(u, 2u; dims = 3)
    t2_aux = SFC.calculate_structure_function_tensor(
        Val(2), x, u_aux, bins; backend = SFC.SerialBackend(),
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
    Test.@test SF.KHM.bin_midpoints([0.0, 2.0, 6.0]) == [1.0, 4.0]
    Test.@test collect(SF.KHM.bin_midpoints(SF.LinearBinEdges(range(0.0, 6.0; length = 4)))) ==
        [1.0, 3.0, 5.0]
    Test.@test SF.KHM.bin_midpoints(SF.BinEdges([0.0, 2.0, 6.0])) == [1.0, 4.0]
end
