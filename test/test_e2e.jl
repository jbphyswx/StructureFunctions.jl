using Test: Test
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

include(joinpath(@__DIR__, "test_synthetic_data.jl"))
using .SyntheticData: SyntheticData

Test.@testset "E2E: Structure Function E2E Suite" begin
    xs, ys, _ = SyntheticData.generate_nonuniform_domain(
        16;
        R_mask = 0.0,
        lat_range = (0.0, 5.0),
        lon_range = (0.0, 10.0),
    )
    dk = 2π / 10.0
    u = real.(SyntheticData.generate_spectral_field(xs, ys; peaks = [(2dk, dk, 1.0)]))
    v = real.(SyntheticData.generate_spectral_field(xs, ys; peaks = [(dk, 2dk, 0.5)]))
    pos = vcat(vec(xs)', vec(ys)')
    vals = vcat(vec(u)', vec(v)')
    r_bins = collect(0.1 .+ (0:5) .* 0.4)
    sf64 = SFC.calculate_structure_function(SFT.SecondOrderStructureFunction, pos, vals, r_bins)

    # Every pair of a 256-point non-uniform domain, binned on [lo, hi) and averaged directly.
    Test.@testset "the second-order structure function is the pair average" begin
        N = size(pos, 2)
        sums, counts = zeros(5), zeros(Int, 5)
        for i in 1:(N - 1), j in (i + 1):N
            b = searchsortedlast(r_bins, sqrt(sum(abs2, pos[:, j] .- pos[:, i])))
            1 <= b <= 5 || continue
            sums[b] += sum(abs2, vals[:, j] .- vals[:, i])
            counts[b] += 1
        end
        Test.@test sf64.values ≈ sums ./ counts rtol = 1e-12
    end

    Test.@testset "Float32 input gives the Float64 result to single precision" begin
        sf32 = SFC.calculate_structure_function(
            SFT.SecondOrderStructureFunction,
            Float32.(pos),
            Float32.(vals),
            Float32.(r_bins),
        )
        Test.@test sf32 ≈ Float32.(sf64) rtol = 1e-5
    end
end
