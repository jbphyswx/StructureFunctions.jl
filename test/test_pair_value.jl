using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using Random: Random

const RAW = SF.StructureFunctionSumsAndCounts

const SERIAL = CB.SerialBackend()

# (operator, FT, D, backend): every operator, D = 4 taking the scalar kernel and D ∈ (2, 3) the SIMD one.
const SUM_CASES = (
    (SFT.L2SF, Float32, 2, SERIAL), (SFT.T2SF, Float64, 3, SERIAL), (SFT.S3SF, Float32, 3, SERIAL),
    (SFT.L3SF, Float64, 2, SERIAL), (SFT.L1T2SF, Float32, 2, SERIAL), (SFT.T2ComponentSF, Float64, 2, SERIAL),
    (SFT.L1T2ComponentSF, Float32, 3, SERIAL), (SFT.T2ComponentSF, Float64, 4, SERIAL),
)

# (FT, D, operators) of the per-pair check: each operator at both float types and both widths.
const FORM_CASES = (
    (Float32, 2, (SFT.L2SF, SFT.T2SF, SFT.S3SF, SFT.L3SF)), (Float64, 3, (SFT.L2SF, SFT.T2SF, SFT.S3SF, SFT.L3SF)),
    (Float32, 3, (SFT.L1T2SF, SFT.T2ComponentSF, SFT.L1T2ComponentSF)),
    (Float64, 2, (SFT.L1T2SF, SFT.T2ComponentSF, SFT.L1T2ComponentSF)),
)

const OPS = (SFT.L2SF, SFT.T2SF, SFT.S3SF, SFT.L3SF, SFT.L1T2SF, SFT.T2ComponentSF, SFT.L1T2ComponentSF)

const PARALLEL_CASES = ((Float32, 1.7f0), (Float32, -2.9f0), (Float64, 0.3))

"""Per-bin `Σ sf` and `Σ |sf|` over every pair, in Float64 from increments formed in the inputs' type."""
function operator_sums(sf, x::AbstractMatrix, u::AbstractMatrix, bins, ::Val{D}) where {D}
    ref = zeros(length(bins) - 1); mag = zeros(length(bins) - 1)
    for i in 1:size(x, 2), j in (i + 1):size(x, 2)
        dx = SA.SVector{D, Float64}(ntuple(d -> Float64(x[d, j] - x[d, i]), Val(D))); r = sqrt(sum(abs2, dx))
        b = searchsortedfirst(bins, r) - 1
        1 <= b < length(bins) || continue
        v = sf(SA.SVector{D, Float64}(ntuple(d -> Float64(u[d, j] - u[d, i]), Val(D))), dx / r)
        ref[b] += v; mag[b] += abs(v)
    end
    return ref, mag
end

Test.@testset "A pair's value" begin
    rng = Random.Xoshiro(20260925)
    Test.@testset "sums are the operator's, $sf, $FT, D = $D, $(nameof(typeof(be)))" for (sf, FT, D, be) in SUM_CASES
        N = 160
        x = rand(rng, FT, D, N); u = randn(rng, FT, D, N)
        bins = collect(range(FT(0), FT(1.2); length = 9))
        ref, mag = operator_sums(sf, x, u, bins, Val(D))
        r0 = SF.to_host(SFC.calculate_structure_function(sf, x, u, bins, RAW; backend = be))
        # Per bin against Σ|v|: an odd moment's bin sum cancels, and the kernels sum in FT.
        Test.@test all(abs.(r0.sums .- ref) .<= (FT == Float32 ? 1e-4 : 1e-10) .* mag)
    end
    Test.@testset "every form of the value agrees" begin
        # Each assertion lists the operators that disagree on any of the drawn pairs.
        for (FT, D, ops) in FORM_CASES
            g = SFH.FlatGeometry{D}()
            tol = FT == Float32 ? 1e-5 : 1e-13
            pairs = [(SA.SVector{D, FT}(randn(rng, FT, D)), SA.SVector{D, FT}(randn(rng, FT, D))) for _ in 1:16]
            agrees(sf, form) = all(pairs) do (dx, du)
                r2 = SFH.fma_dot(dx, dx)
                isapprox(form(sf, du, dx, r2), SFT.pair_value(sf, g, dx, sqrt(r2), du);
                         rtol = tol, atol = tol * (1 + SFH.fma_dot(du, du))^2)
            end
            along_unit(sf, du, dx, r2) = sf(du, dx / sqrt(r2))
            Test.@test (FT, D, filter(sf -> !agrees(sf, along_unit), ops)) == (FT, D, ())
            Test.@test (FT, D, filter(sf -> !agrees(sf, SFT.flat_pair_value), ops)) == (FT, D, ())
        end
        gs = SFH.SphericalGeometry{2}(DI.Haversine(1.0), 1.0)
        spherical = map(1:16) do _
            p1 = SFH.unit_position(360rand(rng), 180rand(rng) - 90)
            p2 = SFH.unit_position(360rand(rng), 180rand(rng) - 90)
            ok, r, frame = SFH.pair_frame(gs, p1, p2)
            (ok, r, frame, SFH.pair_delta(gs, frame, p1, p2, SA.SVector{3}(randn(rng, 3)), SA.SVector{3}(randn(rng, 3))))
        end
        on_sphere(sf) = all(!ok || SFT.pair_value(sf, gs, frame, r, du) ≈ sf(du, SFH.pair_direction(gs, frame, r))
                            for (ok, r, frame, du) in spherical)
        Test.@test filter(sf -> !on_sphere(sf), OPS) == ()
        dx = SA.SVector(0.3, -0.7); v = SA.SVector(0.2, 1.1); θ = -0.4
        inc = MF.FieldIncrement{2, 1, 1, Float64}((v,), (θ,))
        r = sqrt(SFH.fma_dot(dx, dx))
        mixed = (SFT.MixedSFType{1, 0, 2}(), SFT.MixedSFType{1, 2, 1}(), SFT.MixedSFType{0, 1, 1}())
        Test.@test filter(sf -> !(SFT.pair_value(sf, SFH.FlatGeometry{2}(), dx, r, inc) ≈ sf(inc, dx / r)), mixed) == ()
    end
    Test.@testset "an odd transverse power of a parallel increment is finite" begin
        # Float32 at s = 1.7 and -2.9 rounds the transverse energy below zero.
        finite(FT, s) = (dx = SA.SVector{2, FT}(FT(0.1), FT(0.2));
                         inc = MF.FieldIncrement{2, 1, 1, FT}((s * dx,), (FT(0.5),));
                         isfinite(SFT.pair_value(SFT.MixedSFType{0, 1, 1}(), SFH.FlatGeometry{2}(), dx,
                                                 sqrt(SFH.fma_dot(dx, dx)), inc)))
        Test.@test filter(c -> !finite(c...), PARALLEL_CASES) == ()
    end
end
