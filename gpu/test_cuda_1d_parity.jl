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
const NB = 20
const CAPACITY = SFC._val_int(CE._cuda_1d_bin_capacity(NB))
const BINS = collect(FT, range(0.05, 1.5; length = NB + 1))
const GEOM = SF.HelperFunctions.FlatGeometry{D}()
moments(NMOM) = NMOM == 1 ? SFT.L2SFType() : SFT.SinglePassInvariants()
kind(NMOM) = Val(NMOM == 1 ? :sf1d : :single_pass)

plan_kind(::CE.CUDA1DPlan) = :direct
plan_kind(::CE.CUDA1DStripPlan) = :strip

"""`(NMOM, NB, B)` sums and counts of the portable kernels on `KA.CPU()`; `x` is `(D, N)` when `fixed`."""
function reference(x, u, bins, NMOM, fixed)
    nb, b = length(bins) - 1, size(u, 3)
    out, cnt = zeros(FT, NMOM, nb, b), zeros(UInt32, NMOM, nb, b)
    GE._sf_launch_1d_batch!(KA.CPU(), out, cnt, x, u, moments(NMOM), GE._gpu_digitizer(KA.CPU(), bins, kind(NMOM)),
                            size(u, 2), nb, b, fixed, GEOM)
    KA.synchronize(KA.CPU())
    return out, cnt
end

"""Host copies of the `(NMOM, NB, B)` sums and counts of the native launch of `plan`, a plan or a rule."""
function device(plan, x, u, bins, NMOM, fixed)
    nb, b = length(bins) - 1, size(u, 3)
    out, cnt = CUDA.zeros(FT, NMOM, nb, b), CUDA.zeros(UInt32, NMOM, nb, b)
    SFC.gpu_native_launch_1d!(plan, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u), SFC.NoWeights(), moments(NMOM),
                              GE._gpu_digitizer(BE, bins, kind(NMOM)), size(u, 2), nb, b, fixed, GEOM, nothing)
    return Array(out), Array(cnt)
end

# (plan kind, moments, shared positions, plan spec)
const PLANS = (
    (:direct, 1, false, (128, 2)),
    (:direct, 6, true, (256, 1)),
    (:direct, 6, false, (128, 4)),
    (:strip, 1, true, (:strip, 256, 2, 2)),
    (:strip, 6, true, (:strip, 128, 2, 1)),
)

Test.@testset "native 1-D kernels against KA.CPU()" begin
    # Every native 1-D plan kind, launched as planned.
    Test.@testset "$k NMOM=$NMOM shared=$fixed $spec" for (i, (k, NMOM, fixed, spec)) in enumerate(PLANS)
        Random.seed!(20260916 + i)
        x = fixed ? rand(FT, D, N) : rand(FT, D, N, B)
        u = randn(FT, D, N, B)
        plan = CE._cuda_1d_fit(CAPS, FT, FT, FT, UInt32, D, D, NMOM, CAPACITY, spec)
        Test.@test plan_kind(plan) === k
        o, c = device(plan, x, u, BINS, NMOM, fixed)
        ro, rc = reference(x, u, BINS, NMOM, fixed)
        Test.@test c == rc
        Test.@test isapprox(o, ro; rtol = 1e-10)
    end

    # A call over per-slice positions takes the rule's plan for its size.
    Test.@testset "rule" begin
        Random.seed!(20260917)
        x, u = rand(FT, D, 1000, 4), randn(FT, D, 1000, 4)
        rule = CE._cuda_1d_plan(CAPS, FT, FT, FT, UInt32, SFC.NoWeights(), GEOM, NB, moments(6))
        o, c = device(rule, x, u, BINS, 6, false)
        ro, rc = reference(x, u, BINS, 6, false)
        Test.@test c == rc
        Test.@test isapprox(o, ro; rtol = 1e-10)
    end

    # A batch over shared positions that samples its in-range share takes the first candidate, and its class's second
    # call times them all.
    Test.@testset "plan choice" begin
        Random.seed!(20260918)
        n, b = 1000, 4
        Test.@test (n * (n - 1) ÷ 2) * b >= CE.CU_1D_CHOOSE_FROM[2]
        x, u = rand(FT, D, n), randn(FT, D, n, b)
        bins = collect(FT, range(0.0, 0.15; length = NB + 1))
        rule = CE._cuda_1d_plan(CAPS, FT, FT, FT, UInt32, SFC.NoWeights(), GEOM, NB, moments(6))
        ro, rc = reference(x, u, bins, 6, true)
        for _ in 1:2
            o, c = device(rule, x, u, bins, 6, true)
            Test.@test c == rc
            Test.@test isapprox(o, ro; rtol = 1e-10)
        end
        Test.@test !isempty(rule.fixed.chosen)
    end
end
