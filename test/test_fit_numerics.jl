using Test: Test
using LinearAlgebra: LinearAlgebra
using LsqFit: LsqFit
using StructureFunctions: Calculations as C

Test.@testset "Stable linear fits" begin
    # The second singular direction is resolvable in Float64 but disappears in H'H.
    H = [1.0 1.0; 1.0 1.0 + 1e-8; 1.0 1.0 - 1e-8]
    truth = [2.0, -1.0]
    x, covariance = C.solve(C.RegularizedLeastSquares(nothing), H, H * truth, ones(3))
    Test.@test x ≈ truth rtol=5e-8
    Test.@test all(isfinite, covariance)
    # The dense covariance has condition number near 1/eps(); its small
    # eigenvalue is not representable reliably. Compare entries to BigFloat.
    Hbig = BigFloat.(H)
    Cbig = (Hbig' * Hbig) \ Matrix{BigFloat}(I, 2, 2)
    Test.@test covariance ≈ Float64.(Cbig) rtol=5e-8
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), ones(3, 2), ones(3), ones(3))
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), ones(1, 2), ones(1), ones(1))

    # Correlated prior and data: compare with a high-precision independent solve.
    H = [1.0 2; 3 1; 0 2]
    y = [3.0, -1, 2]
    W = [2.0 0.1 0; 0.1 1 0.2; 0 0.2 3]
    P = [2.0 0.4; 0.4 1]
    x, covariance = C.solve(C.RegularizedLeastSquares(P), H, y, W)
    Hb, Wb, Pb, yb = BigFloat.(H), BigFloat.(W), BigFloat.(P), BigFloat.(y)
    precision = Hb' * (Wb \ Hb) + Pb \ Matrix{BigFloat}(I, 2, 2)
    Test.@test x ≈ Float64.(precision \ (Hb' * (Wb \ yb))) rtol=2e-14
    Test.@test covariance ≈ Float64.(precision \ Matrix{BigFloat}(I, 2, 2)) rtol=2e-14
    Test.@test isposdef(covariance)
    for bad in ([1.0, Inf, 1.0], [1.0, NaN, 1.0], [1.0, 0.0, 1.0])
        Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), H, y, bad)
    end
    Test.@test_throws ArgumentError C.solve(C.RegularizedLeastSquares(nothing), H, y, [1.0 1 0; 0 1 0; 0 0 1])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0, Inf])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0 2; 0 1])
    Test.@test_throws ArgumentError C.RegularizedLeastSquares([1.0 2; 2 1])
    Test.@test_throws DimensionMismatch C.solve(C.RegularizedLeastSquares(nothing), H, y[1:2], ones(2))
end

Test.@testset "NNLS termination and optimality" begin
    for scale in (1e-12, 1.0, 1e12)
        A = [1.0 0 1; 0 1 1; 1 1 2] # dependent columns
        b = scale .* [1.0, -1, 0]
        info = C._nnls(A, b; return_info=true)
        Test.@test info.converged
        Test.@test info.iterations <= 30
        Test.@test all(>=(0), info.x)
        gradient = A' * (A * info.x - b)
        Test.@test minimum(gradient) >= -1e-12 * scale
        Test.@test maximum(abs, info.x .* gradient) <= 1e-12 * scale^2
    end
    info = C._nnls(Matrix{Float64}(I, 2, 2), ones(2); maxiter=0, return_info=true)
    Test.@test !info.converged
    Test.@test info.iterations == 0
    Test.@test_throws ErrorException C._nnls(Matrix{Float64}(I, 2, 2), ones(2); maxiter=0)
    Test.@test C._nnls(zeros(3, 2), ones(3)) == zeros(2)
    Test.@test_throws ArgumentError C._nnls([1.0 NaN], [1.0])
end

Test.@testset "Relative fitting with zero observations" begin
    Test.@test C._relative_scales(zeros(3)) == ones(3)
    Test.@test all(>(0), C._relative_scales([0.0, 1e-20, 2e-20]))
    Test.@test_throws ArgumentError C._relative_scales([NaN])
    r = collect(range(0.1, 1.0; length=8))
    p, covariance, edges, converged = C._segmented_fit(C.SegmentedPowerLaw(1), Val(1), r, zeros(8), nothing, 0.2, 2.0)
    Test.@test all(isfinite, p)
    Test.@test all(isfinite, covariance)
    Test.@test p[1] < 1e-12
    x, covariance, info = C.solve(C.NonNegativeLeastSquares(), Matrix{Float64}(I, 2, 2), ones(2), nothing;
                                  maxiter=0, return_info=true)
    Test.@test !info.converged
    Test.@test covariance === nothing
    Test.@test_throws ErrorException C.solve(C.NonNegativeLeastSquares(), Matrix{Float64}(I, 2, 2), ones(2), nothing; maxiter=0)
end
