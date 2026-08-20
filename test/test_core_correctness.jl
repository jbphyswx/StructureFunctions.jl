using StructureFunctions:
    StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC
using Test: Test
using Random: Random
using StaticArrays: StaticArrays as SA

Test.@testset "Core Correctness - Block A" begin
    Test.@testset "Blocked path regression" begin
        x = [0.0 1.0; 0.0 0.0]
        u = [1.0 2.0; 0.0 0.0]
        bins = SA.SVector(0.0, 2.0)
        sf_type = SFT.LongitudinalSecondOrderStructureFunction
        Test.@test_nowarn SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )
    end

    Test.@testset "Pair-count verification" begin
        # N=3 points -> N(N-1)/2 = 3 pairs
        x = [0.0 1.0 2.0; 0.0 0.0 0.0]
        u = [0.0 0.0 0.0; 0.0 0.0 0.0]
        bins = SA.SVector(0.0, 3.0)
        sf_type = SFT.SecondOrderStructureFunction

        res = SFC.calculate_structure_function(sf_type, x, u, bins;
            verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts)
        Test.@test sum(res.counts) == 3

        # N=4 points -> 4*3/2 = 6 pairs
        x4 = [0.0 1.0 2.0 3.0; 0.0 0.0 0.0 0.0]
        res4 = SFC.calculate_structure_function(sf_type, x4, zeros(2, 4), bins;
            verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts)
        Test.@test sum(res4.counts) == 6
    end

    Test.@testset "Numerical reference (Tiny case)" begin
        # 2 points: (0,0) and (1,0)
        # Velocities: (1,0) and (2,0)
        # du = (1, 0), rhat = (1, 0)
        # SecondOrderLongitudinal: (du . rhat)^2 = 1^2 = 1.0
        x = [0.0 1.0; 0.0 0.0]
        u = [1.0 2.0; 0.0 0.0]
        bins = SA.SVector(0.0, 2.0)
        sf_type = SFT.LongitudinalSecondOrderStructureFunction

        val = SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )
        Test.@test val[1] ≈ 1.0

        # 3 points: (0,0), (1,0), (0,1)
        # Velocities: (0,0), (1,0), (0,1)
        # Pairs:
        # 1-2: dx=(1,0), du=(1,0), rhat=(1,0), du.rhat=1, (du.rhat)^2 = 1
        # 1-3: dx=(0,1), du=(0,1), rhat=(0,1), du.rhat=1, (du.rhat)^2 = 1
        # 2-3: dx=(-1,1), du=(-1,1), rhat=(-1,1)/sqrt(2), du.rhat=sqrt(2), (du.rhat)^2 = 2
        # Sum = 1 + 1 + 2 = 4
        # Count = 3
        # Mean = 4/3
        x3 = [0.0 1.0 0.0; 0.0 0.0 1.0]
        u3 = [0.0 1.0 0.0; 0.0 0.0 1.0]
        val3 = SFC.calculate_structure_function(
            sf_type,
            x3,
            u3,
            bins;
            verbose = false,
            show_progress = false,
        )
        Test.@test val3[1] ≈ 4 / 3
    end

    Test.@testset "SF wiring and signed magnitude consistency" begin
        # Test case: 2 points at (0,0) and (1,0) -> rhat = (1, 0), nhat = (0, -1)
        # Velocities: (0,1) and (0,2) -> du = (0,1) [Transverse]
        x = [0.0 1.0; 0.0 0.0]
        u = [0.0 0.0; 1.0 2.0]
        bins = SA.SVector(0.0, 2.0)

        # Longitudinal Second Order: (du.rhat)^2 = 0^2 = 0
        Test.@test SFC.calculate_structure_function(
            SFT.LongitudinalSecondOrderStructureFunction,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )[1][1] == 0.0

        # Transverse Second Order: |du_t|^2 = (-1)^2 = 1
        Test.@test SFC.calculate_structure_function(
            SFT.TransverseSecondOrderStructureFunction,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )[1][1] == 1.0

        # Diagonal Consistent Third Order (l^3): 0^3 = 0
        Test.@test SFC.calculate_structure_function(
            SFT.DiagonalConsistentThirdOrderStructureFunction,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )[1][1] == 0.0

        # Off-Diagonal Consistent Third Order (t^3): with the right-handed n̂ = ẑ × r̂ the transverse
        # component is +1 here, so t^3 = +1 (it was -1 under the old clockwise n̂).
        Test.@test SFC.calculate_structure_function(
            SFT.OffDiagonalConsistentThirdOrderStructureFunction,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )[1][1] == 1.0

        # Off-Diagonal Inconsistent Third Order (l*t^2): 0 * (-1)^2 = 0
        Test.@test SFC.calculate_structure_function(
            SFT.OffDiagonalInconsistentThirdOrderStructureFunction,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )[1][1] == 0.0
    end

    Test.@testset "3D invariant transverse and S3 semantics" begin
        δu = SA.SVector(2.0, 3.0, 4.0)
        r̂ = SA.SVector(1.0, 0.0, 0.0)

        Test.@test SFT.S2SF(δu, r̂) ≈ SFT.L2SF(δu, r̂) + SFT.T2SF(δu, r̂)
        Test.@test SFT.L2SF(δu, r̂) ≈ 4.0
        Test.@test SFT.T2SF(δu, r̂) ≈ 25.0
        Test.@test SFT.T2ComponentSF(δu, r̂) ≈ 12.5
        Test.@test SFT.L3SF(δu, r̂) ≈ 8.0
        Test.@test SFT.L1T2SF(δu, r̂) ≈ 50.0
        Test.@test SFT.L1T2ComponentSF(δu, r̂) ≈ 25.0
        Test.@test SFT.S3SF(δu, r̂) ≈ SFT.L3SF(δu, r̂) + SFT.L1T2SF(δu, r̂)
        Test.@test SFT.S3SF(δu, r̂) != SFT.FullVectorStructureFunctionType(3)(δu, r̂)
        Test.@test SFT.FullVectorStructureFunctionType(3)(δu, r̂) ≈ sqrt(sum(abs2, δu))^3
    end

    Test.@testset "Signed transverse operators use the documented 2D orientation" begin
        δu = SA.SVector(2.0, 3.0)
        r̂ = SA.SVector(1.0, 0.0)

        # n̂ = ẑ × r̂ = (0, 1) is the right-handed quarter turn, so δu_T = +3. These two operators are
        # the ONLY ones whose sign depends on that choice; every other operator consumes δu_T².
        Test.@test SFT.L2T1SF(δu, r̂) ≈ 12.0
        Test.@test SFT.T3SF(δu, r̂) ≈ 27.0

        # Orientation is a property of the convention, not of this fixture: rotating the pair must
        # leave both invariant, and reflecting it must flip exactly these two.
        θ = 0.7
        Rot = SA.SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        Test.@test SFT.L2T1SF(Rot * δu, Rot * r̂) ≈ SFT.L2T1SF(δu, r̂)
        Test.@test SFT.T3SF(Rot * δu, Rot * r̂) ≈ SFT.T3SF(δu, r̂)
        Flip = SA.SMatrix{2, 2}(1.0, 0.0, 0.0, -1.0)
        Test.@test SFT.L2T1SF(Flip * δu, Flip * r̂) ≈ -SFT.L2T1SF(δu, r̂)
        Test.@test SFT.T3SF(Flip * δu, Flip * r̂) ≈ -SFT.T3SF(δu, r̂)
        Test.@test SFT.T2SF(Flip * δu, Flip * r̂) ≈ SFT.T2SF(δu, r̂)
    end
end

Test.@testset "Type Stability and Performance - Block C" begin
    x = [0.0 1.0 2.0; 0.0 0.0 0.0]
    u = [1.0 2.0 3.0; 0.0 0.0 0.0]
    bins = SA.SVector(0.0, 3.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    # Test inference of the core kernel call
    δu = SA.SVector{2, Float64}(1.0, 0.0)
    r̂ = SA.SVector{2, Float64}(1.0, 0.0)
    Test.@test Test.@inferred(sf_type(δu, r̂)) == 1.0

    # Test inference of the calculator
    Test.@testset "Inference of calculate_structure_function" begin
        # We manually check the type to account for the intentional Union return
        # from the output_type keyword.
        res = Test.@test_nowarn SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )
        Test.@test res isa SF.AbstractStructureFunction

        # We check that the inner logic is stable enough
        Test.@test typeof(res) <: SF.StructureFunction
    end

    Test.@testset "Allocation check" begin
        # Baseline check
        Test.@test_nowarn SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )
    end
end

# `count_eltype` defaults to UInt32 everywhere (device histograms are UInt32 for shared-memory
# reasons). Every pair can land in one bin, so past N = 92682 an unsigned counter wraps silently and
# `_bin_average` then divides by the wrapped value — a plausible, wrong mean. The bound is checked.
Test.@testset "Count element type must represent the worst-case pair count" begin
    Test.@test SFC._assert_counts_representable(UInt32, 92682) === nothing   # 4_294_930_821 pairs
    Test.@test_throws ArgumentError SFC._assert_counts_representable(UInt32, 92683)
    Test.@test SFC._assert_counts_representable(UInt64, 1_000_000) === nothing
    Test.@test SFC._assert_counts_representable(Int64, 1_000_000) === nothing

    bins_of = collect(Float32, range(0.0f0, 2.0f0; length = 9))
    sft = SFT.L2SFType()
    big_x = zeros(Float32, 2, 100_000)

    # Fires at the public boundary, before any O(N^2) work is done.
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        sft, big_x, big_x, bins_of; verbose = false, show_progress = false,
    )
    # A caller-supplied counts buffer is validated on its own element type.
    Test.@test_throws ArgumentError SFC.calculate_structure_function!(
        zeros(Float32, 8), zeros(UInt32, 8), sft, big_x, big_x, bins_of;
        verbose = false, show_progress = false,
    )
    # Small problems are unaffected.
    small_x = rand(Float32, 2, 64)
    Test.@test SFC.calculate_structure_function(
        sft, small_x, small_x, bins_of; verbose = false, show_progress = false,
    ) isa Any
end

# The auto-binning min/max scan covers unordered pairs once (`j > i`) and accumulates in the input
# eltype; both are silent if wrong — the bins would just come out slightly different.
Test.@testset "Auto-binning min/max scan" begin
    Random.seed!(808)
    for FT in (Float64, Float32)
        x = rand(FT, 2, 250) .* FT(10)
        mn, mx = SFC._minmax_matrix_for_autobins(x, SFC.DI.Euclidean(), false)

        # brute force over every unordered pair
        bmn, bmx = FT(Inf), FT(0)
        for i in 1:size(x, 2), j in (i + 1):size(x, 2)
            d = sqrt((x[1, j] - x[1, i])^2 + (x[2, j] - x[2, i])^2)
            bmn = min(bmn, d); bmx = max(bmx, d)
        end
        Test.@test mn ≈ bmn
        Test.@test mx ≈ bmx
        # Float64 seed literals here would silently widen the bin edges for Float32 input.
        Test.@test typeof(mn) === FT
        Test.@test typeof(mx) === FT
    end

    # Auto-binned edges stay in the input precision end to end.
    x32 = rand(Float32, 2, 200); u32 = randn(Float32, 2, 200)
    r = SFC.calculate_structure_function(
        SFT.L2SFType(), x32, u32, 12; verbose = false, show_progress = false,
    )
    Test.@test eltype(r.distance) === Float32
    Test.@test eltype(r.values) === Float32
end
