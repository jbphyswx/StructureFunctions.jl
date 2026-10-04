using Test: Test
using CUDA: CUDA
import KernelAbstractions as KA
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const CE = Base.get_extension(SF, :StructureFunctionsCUDAExt)
const BE = CUDA.CUDABackend()
const CAPS = SFC.gpu_device_caps(BE)
const FT = Float64
const D = 2
const N = 300
const B = 3
const GEOM = SF.HelperFunctions.FlatGeometry{D}()
const AXIS = SFC.InvariantValueAxis()
moments(NMOM) = NMOM == 1 ? SFT.SecondOrderStructureFunctionType() : SFT.SinglePassInvariants()
kind(NMOM) = Val(NMOM == 1 ? :joint2d : :single_pass_2d)
dist_bins(n) = collect(FT, range(0.05, 2.0; length = n + 1))
value_bins(n) = collect(FT, range(-5.0, 5.0; length = n + 1))

plan_kind(::CE.CUDA2DPlan{W, F, M, T, C, NP}) where {W, F, M, T, C, NP} = NP == M ? :shared : :planes
plan_kind(::CE.CUDA2DGlobalPlan) = :global

"""`(NMOM, n_dist, n_val, B)` sums and counts of the portable kernels on `KA.CPU()`; `x` is `(D, N)` when `fixed`."""
function reference(x, u, bins, vb, NMOM, fixed)
    nd, nv, b = length(bins) - 1, length(vb) - 1, size(u, 3)
    out, cnt = zeros(FT, NMOM, nd, nv, b), zeros(UInt32, NMOM, nd, nv, b)
    GE._sf_launch_2d_batch!(KA.CPU(), out, cnt, x, u, moments(NMOM), GE._gpu_digitizer(KA.CPU(), bins, kind(NMOM)),
                            GE._gpu_digitizer(KA.CPU(), vb, Val(:value)), size(u, 2), nd, nv, b, fixed, GEOM, AXIS)
    KA.synchronize(KA.CPU())
    return out, cnt
end

"""Host copies of the native launch of `plan`, a plan or a plan choice whose portable candidate is `portable!`."""
function device(plan, x, u, bins, vb, NMOM, fixed)
    n, nd, nv, b = size(u, 2), length(bins) - 1, length(vb) - 1, size(u, 3)
    out, cnt = CUDA.zeros(FT, NMOM, nd, nv, b), CUDA.zeros(UInt32, NMOM, nd, nv, b)
    xd, ud = CUDA.CuArray(x), CUDA.CuArray(u)
    ddig, vplan = GE._gpu_digitizer(BE, bins, kind(NMOM)), GE._gpu_digitizer(BE, vb, Val(:value))
    portable!(o, c) = GE._sf_launch_2d_batch_portable!(BE, o, c, xd, ud, moments(NMOM), ddig, vplan, n, nd, nv, b,
                                                       fixed, GEOM, AXIS)
    SFC.gpu_native_launch_2d!(plan, out, cnt, xd, ud, SFC.NoWeights(), moments(NMOM), ddig, vplan, n, nd, nv, b, fixed,
                              GEOM, AXIS, nothing, portable!)
    return Array(out), Array(cnt)
end

# (plan kind, moments, shared positions, plan spec, distance bins, value bins)
const PLANS = (
    (:shared, 1, true, (128, 1), 16, 8),
    (:shared, 6, false, (128, 6), 30, 30),
    (:planes, 6, true, (128, 2), 16, 8),
    (:global, 1, false, (128, 0), 16, 8),
    (:global, 6, true, (128, 0), 20, 20),
)

Test.@testset "native 2-D kernels against KA.CPU()" begin
    # Every native 2-D plan kind, launched as planned.
    Test.@testset "$k NMOM=$NMOM shared=$fixed $(nd)x$(nv)" for (i, (k, NMOM, fixed, spec, nd, nv)) in enumerate(PLANS)
        Random.seed!(20260916 + i)
        x = fixed ? rand(FT, D, N) : rand(FT, D, N, B)
        u = randn(FT, D, N, B)
        bins, vb = dist_bins(nd), value_bins(nv)
        plan = CE._cuda_2d_fit(CAPS, FT, FT, FT, UInt32, D, D, NMOM, nd, nv, spec)
        Test.@test plan_kind(plan) === k
        o, c = device(plan, x, u, bins, vb, NMOM, fixed)
        ro, rc = reference(x, u, bins, vb, NMOM, fixed)
        Test.@test c == rc
        Test.@test isapprox(o, ro; rtol = 1e-10)
    end

    # A call that samples its in-range share takes the first candidate, and its class's second call times them all.
    Test.@testset "plan choice" begin
        Random.seed!(20260917)
        n, b, nd, nv = 1000, 4, 16, 8
        Test.@test (n * (n - 1) ÷ 2) * b >= CE.CU_2D_CHOOSE_FROM[2]
        x, u = rand(FT, D, n), randn(FT, D, n, b)
        bins, vb = dist_bins(nd), value_bins(nv)
        choice = CE._cuda_2d_plan(CAPS, FT, FT, FT, UInt32, SFC.NoWeights(), GEOM, moments(6), nd, nv)
        ro, rc = reference(x, u, bins, vb, 6, true)
        for _ in 1:2
            o, c = device(choice, x, u, bins, vb, 6, true)
            Test.@test c == rc
            Test.@test isapprox(o, ro; rtol = 1e-10)
        end
        Test.@test !isempty(choice.chosen)
    end
end
