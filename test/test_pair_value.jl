module TestPairValue

using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using Random: Random

const RAW = SF.StructureFunctionSumsAndCounts
kw(be) = (; backend = be)

Test.@testset "A pair's value" begin
    rng = Random.Xoshiro(20260925)
    backends = (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
    ops = (SFT.L2SF, SFT.T2SF, SFT.S3SF, SFT.L3SF, SFT.L1T2SF, SFT.T2ComponentSF, SFT.L1T2ComponentSF)
    for FT in (Float32, Float64), D in (2, 3, 4)
        N = 300
        x = rand(rng, FT, D, N); u = randn(rng, FT, D, N)
        bins = collect(range(FT(0), FT(1.2); length = 9))
        Test.@testset "sums are the operator's, $sf, $FT, D = $D" for sf in ops
            ref = zeros(length(bins) - 1); mag = zeros(length(bins) - 1)
            for i in 1:N, j in (i + 1):N
                dx = Float64.(SA.SVector{D}(x[:, j] - x[:, i])); r = sqrt(sum(abs2, dx))
                b = searchsortedfirst(bins, r) - 1
                1 <= b < length(bins) || continue
                v = sf(SA.SVector{D, Float64}(u[:, j] - u[:, i]), dx / r)
                ref[b] += v; mag[b] += abs(v)
            end
            for be in backends
                r0 = SF.to_host(SFC.calculate_structure_function(sf, x, u, bins, UInt64, RAW; kw(be)...))
                # Per bin against Σ|v|: an odd moment's bin sum cancels, and the kernels sum in FT.
                Test.@test all(abs.(r0.sums .- ref) .<= (FT == Float32 ? 1e-4 : 1e-10) .* mag)
            end
        end
    end
    Test.@testset "every form of the value agrees" begin
        for FT in (Float32, Float64), D in (2, 3)
            g = SFH.FlatGeometry{D}()
            tol = FT == Float32 ? 1e-5 : 1e-13
            for _ in 1:200, sf in ops
                dx = SA.SVector{D, FT}(randn(rng, FT, D)); du = SA.SVector{D, FT}(randn(rng, FT, D))
                r2 = SFH.fma_dot(dx, dx); r = sqrt(r2)
                v = SFT.pair_value(sf, g, dx, r, du)
                scale = tol * (1 + SFH.fma_dot(du, du))^2
                Test.@test isapprox(sf(du, dx / r), v; rtol = tol, atol = scale)
                Test.@test isapprox(SFT.flat_pair_value(sf, du, dx, r2), v; rtol = tol, atol = scale)
            end
        end
        gs = SFH.SphericalGeometry{2}(DI.Haversine(1.0), 1.0)
        for _ in 1:200
            p1 = SFH.unit_position(360rand(rng), 180rand(rng) - 90)
            p2 = SFH.unit_position(360rand(rng), 180rand(rng) - 90)
            ok, r, frame = SFH.pair_frame(gs, p1, p2)
            ok || continue
            du = SFH.pair_delta(gs, frame, p1, p2, SA.SVector{3}(randn(rng, 3)), SA.SVector{3}(randn(rng, 3)))
            rh = SFH.pair_direction(gs, frame, r)
            for sf in ops
                Test.@test SFT.pair_value(sf, gs, frame, r, du) ≈ sf(du, rh)
            end
        end
        dx = SA.SVector(0.3, -0.7); v = SA.SVector(0.2, 1.1); θ = -0.4
        inc = MF.FieldIncrement{2, 1, 1, Float64}((v,), (θ,))
        for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.MixedSFType{1, 2, 1}(), SFT.MixedSFType{0, 1, 1}())
            Test.@test SFT.pair_value(sf, SFH.FlatGeometry{2}(), dx, sqrt(SFH.fma_dot(dx, dx)), inc) ≈
                       sf(inc, dx / sqrt(SFH.fma_dot(dx, dx)))
        end
    end
    Test.@testset "an odd transverse power of a parallel increment is finite" begin
        for FT in (Float32, Float64), s in (FT(0.3), FT(1.7), FT(-2.9))
            dx = SA.SVector{2, FT}(FT(0.1), FT(0.2)); v = s * dx
            inc = MF.FieldIncrement{2, 1, 1, FT}((v,), (FT(0.5),))
            r = sqrt(SFH.fma_dot(dx, dx))
            Test.@test isfinite(SFT.pair_value(SFT.MixedSFType{0, 1, 1}(), SFH.FlatGeometry{2}(), dx, r, inc))
        end
    end
end

end # module
