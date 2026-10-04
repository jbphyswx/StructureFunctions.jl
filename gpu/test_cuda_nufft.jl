using Test: Test
using CUDA: CUDA
using Random: Random
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const RAW = SF.StructureFunctionSumsAndCounts
const PROVIDERS = (SFC.NonuniformFFTsSpectralBackend(), SFC.FINUFFTSpectralBackend())

rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / (maximum(abs, Array(b)) + eps())

# Each provider through the public entries on the device, and one plan kept for every slice of a batch on fixed points.
Test.@testset "non-uniform FFT route on the device" begin
    Random.seed!(20260926)
    N, nt = 2000, 3
    s = SFC.ScatteredModesSchedule(rand(2, N), 0.3, (128, 96); taper = SF.GaussianTaper(0.01))
    bins = collect(range(0.0, 0.3; length = 9))
    nb = length(bins) - 1
    u = randn(2, N)
    ub = randn(2, N, nt)
    Test.@testset "$(nameof(typeof(tag)))" for tag in PROVIDERS
        Test.@testset "$(nameof(typeof(sf)))" for sf in (SFT.L2SFType(), SFT.L3SFType())
            ref = SFC.calculate_structure_function(sf, s, u, bins, tag, RAW; backend = SER)
            got = SFC.calculate_structure_function(sf, s, u, bins, tag, RAW; backend = DEV)
            Test.@test got.sums isa CUDA.CuArray
            Test.@test rel(got.sums, ref.sums) < 1e-9
            Test.@test rel(got.counts, ref.counts) < 1e-9
        end
        rs, rc = zeros(nb, nt), zeros(nb, nt)
        SFC.calculate_structure_function_batch!(rs, rc, SFT.L2SFType(), s, ub, bins, tag; backend = SER)
        ws = SFC.TransformWorkspace()
        gs, gc = CUDA.zeros(Float64, nb, nt), CUDA.zeros(Float64, nb, nt)
        SFC.calculate_structure_function_batch!(gs, gc, SFT.L2SFType(), s, ub, bins, tag; backend = DEV, workspace = ws)
        plans = [last(e[2]) for e in ws.pool if e[1] in (:nonuniformffts, :finufft)]
        Test.@test length(plans) == 1
        Test.@test rel(gs, rs) < 1e-9
        Test.@test rel(gc, rc) < 1e-9
        SFC._release_plans!(ws)
    end
end
