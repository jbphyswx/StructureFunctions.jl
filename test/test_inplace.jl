using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

"""Zeroed buffers, the mutating call adding into them, and the serial allocating result, of one input form."""
function ip_form(form, x, u, ub, bins, vbins)
    nb, nv = length(bins) - 1, length(vbins) - 1
    sft, ser = SFT.L2SFType(), CB.SerialBackend()
    if form === :point
        return (zeros(nb), zeros(UInt32, nb)),
               (s, c, be) -> SFC.calculate_structure_function!(s, c, sft, x, u, bins; backend = be),
               SFC.calculate_structure_function(sft, x, u, bins, SF.StructureFunctionSumsAndCounts; backend = ser)
    elseif form === :joint
        return (zeros(nb, nv), zeros(UInt32, nb, nv)),
               (s, c, be) -> SFC.calculate_structure_function!(s, c, sft, x, u, bins, vbins; backend = be),
               SFC.calculate_structure_function(sft, x, u, bins, vbins; backend = ser)
    elseif form === :batch
        return (zeros(nb, size(ub, 3)), zeros(UInt32, nb, size(ub, 3))),
               (s, c, be) -> SFC.calculate_structure_function!(s, c, sft, x, ub, bins; backend = be),
               SFC.calculate_structure_function(sft, x, ub, bins, SF.StructureFunctionSumsAndCounts; backend = ser)
    else
        return (zeros(2, 2, nb), zeros(UInt32, nb)),
               (s, c, be) -> SFC.calculate_structure_function_tensor!(s, c, Val(2), x, u, bins; backend = be),
               SFC.calculate_structure_function_tensor(Val(2), x, u, bins, SFO.StructureFunctionTensorSumsAndCounts;
                                                       backend = ser)
    end
end

# A mutating entry adds exactly the allocating entry's sums and counts into its buffers, on every call; the parallel
# backends' in-place methods add in test_parallel_equivalence.jl.
Test.@testset "a mutating entry accumulates the allocating entry's result" begin
    Random.seed!(1234)
    N = 40
    x, u, ub = rand(2, N), randn(2, N), randn(2, N, 3)
    bins = [0.0, 0.25, 0.5, 1.0, 1.5]
    vbins = collect(range(-1.0, 1.0; length = 11))
    for form in (:point, :joint, :batch, :tensor)
        (s, c), add!, ref = ip_form(form, x, u, ub, bins, vbins)
        add!(s, c, CB.SerialBackend())
        Test.@test s ≈ ref.sums && c == ref.counts
        add!(s, c, CB.SerialBackend())
        Test.@test s ≈ 2 .* ref.sums && c == 2 .* ref.counts
    end
end
