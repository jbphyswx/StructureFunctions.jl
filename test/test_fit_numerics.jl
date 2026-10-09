using Test: Test
using LinearAlgebra: LinearAlgebra
using LsqFit: LsqFit
using StructureFunctions: StructureFunctions as SF, Calculations as C, StructureFunctionTypes as SFT

# An ill-conditioned system, and a correlated prior and data covariance, against BigFloat normal equations.
Test.@testset "regularised least squares is the exact posterior" begin
    H = [1.0 1.0; 1.0 1.0 + 1e-8; 1.0 1.0 - 1e-8]
    truth = [2.0, -1.0]
    x, covariance = C.solve(C.RegularizedLeastSquares(nothing), H, H * truth, ones(3))
    Test.@test x ≈ truth rtol=5e-8
    Hbig = BigFloat.(H)
    Test.@test covariance ≈ Float64.((Hbig' * Hbig) \ Matrix{BigFloat}(LinearAlgebra.I, 2, 2)) rtol=5e-8

    H = [1.0 2; 3 1; 0 2]
    y = [3.0, -1, 2]
    W = [2.0 0.1 0; 0.1 1 0.2; 0 0.2 3]
    P = [2.0 0.4; 0.4 1]
    x, covariance = C.solve(C.RegularizedLeastSquares(P), H, y, W)
    Hb, Wb, Pb, yb = BigFloat.(H), BigFloat.(W), BigFloat.(P), BigFloat.(y)
    precision = Hb' * (Wb \ Hb) + Pb \ Matrix{BigFloat}(LinearAlgebra.I, 2, 2)
    Test.@test x ≈ Float64.(precision \ (Hb' * (Wb \ yb))) rtol=2e-14
    Test.@test covariance ≈ Float64.(precision \ Matrix{BigFloat}(LinearAlgebra.I, 2, 2)) rtol=2e-14
end

Test.@testset "regularised least squares refuses what it cannot solve" begin
    H = [1.0 2; 3 1; 0 2]
    y = [3.0, -1, 2]
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), ones(3, 2), ones(3), ones(3))
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), ones(1, 2), ones(1), ones(1))
    for bad in ([1.0, Inf, 1.0], [1.0, NaN, 1.0], [1.0, 0.0, 1.0])
        Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), H, y, bad)
    end
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), H, y, [1.0 1 0; 0 1 0; 0 0 1])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0, Inf])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0 2; 0 1])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0 2; 2 1])
    Test.@test_throws DimensionMismatch C.solve(C.RegularizedLeastSquares(nothing), H, y[1:2], ones(2))
end

# Dependent columns, and a system whose solve drops a variable it took in, at three scales meet the KKT conditions;
# an unfinished solve is reported or raises.
Test.@testset "non-negative least squares is optimal or says it is not" begin
    systems = (([1.0 0 1; 0 1 1; 1 1 2], [1.0, -1, 0]), ([2.0 0 2; -2 -2 1; 2 0 -1], [3.0, -3, -2]))
    sols = [begin
                b = scale .* b1
                x, _, info = C.solve(C.NonNegativeLeastSquares(), A, b, nothing; return_info = true)
                (; scale, x, info, gradient = A' * (A * x - b))
            end for (A, b1) in systems, scale in (1e-12, 1.0, 1e12)]
    # More solves than final support means a variable left the support; the exact answer is (0, 23/10, 8/5).
    Test.@test all(s -> s.info.iterations > count(>(0), s.x) && s.x ≈ s.scale .* [0, 2.3, 1.6], sols[2, :])
    Test.@test all(s -> s.info.converged, sols)
    Test.@test all(s -> all(>=(0), s.x) && minimum(s.gradient) >= -1e-12 * s.scale, sols)
    Test.@test all(s -> maximum(abs, s.x .* s.gradient) <= 1e-12 * s.scale^2, sols)
    Test.@test C.solve(C.NonNegativeLeastSquares(), zeros(3, 2), ones(3), nothing)[1] == zeros(2)
    Test.@test_throws ArgumentError C.solve(C.NonNegativeLeastSquares(), [1.0 NaN], [1.0], nothing)

    I2 = Matrix{Float64}(LinearAlgebra.I, 2, 2)
    _, _, info = C.solve(C.NonNegativeLeastSquares(), I2, ones(2), nothing; maxiter = 0, return_info = true)
    Test.@test !info.converged
    Test.@test_throws ErrorException C.solve(C.NonNegativeLeastSquares(), I2, ones(2), nothing; maxiter = 0)
end

# All-zero observations fit to a zero amplitude with a finite covariance.
Test.@testset "a segmented fit of zero observations" begin
    edges = collect(range(0.05, 1.05; length = 9))
    f = C.fit_spectrum(SF.StructureFunction(SFT.S2SFType(), edges, zeros(8)), [0.2, 2.0], C.SegmentedPowerLaw(1), Val(1))
    Test.@test all(isfinite, f.parameters) && all(isfinite, f.covariance)
    Test.@test f.parameters[1] < 1e-12
end
