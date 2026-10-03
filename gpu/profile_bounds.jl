# One representative call of a device route: timed after a warm-up, then run once inside the profiler range, so
# Nsight Compute started with `--profile-from-start off` reports only that call's kernels, e.g.
#   ncu --target-processes all --profile-from-start off --set full julia --project=gpu gpu/profile_bounds.jl sf1d
using CUDA: CUDA
using Random: Random
using Printf: Printf
using FFTW: FFTW
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StructureFunctions: LinearBinEdges
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const RAW = SF.StructureFunctionSumsAndCounts
const RNG = Random.Xoshiro(20260926)
const N = 20_000

points(D) = (CUDA.CuArray(rand(RNG, Float32, D, N)), CUDA.CuArray(randn(RNG, Float32, D, N)))
bins(r, n) = LinearBinEdges(0.0f0, Float32(r), n + 1)

const ROUTES = Dict(
    "sf1d" => ("_cuda_sf_1d_", () -> begin
        x, u = points(2)
        () -> SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins(0.3, 32), RAW; backend = DEV)
    end),
    "sp1d" => ("_cuda_sf_1d_", () -> begin
        x, u = points(2)
        () -> SFC.calculate_structure_functions_single_pass(x, u, bins(0.3, 32), RAW; backend = DEV)
    end),
    "joint" => ("_cuda_sf_2d_", () -> begin
        x, u = points(2)
        () -> SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins(0.3, 32), bins(3.0, 20),
                                               SF.StructureFunction2DSumsAndCounts; backend = DEV)
    end),
    "sp2d" => ("_cuda_sf_2d_", () -> begin
        x, u = points(2)
        () -> SFC.calculate_structure_functions_single_pass_2d(x, u, bins(0.3, 50), bins(3.0, 50); backend = DEV)
    end),
    "batch" => ("_cuda_sf_1d_", () -> begin
        x = CUDA.CuArray(rand(RNG, Float32, 2, N))
        u = CUDA.CuArray(randn(RNG, Float32, 2, N, 32))
        s, c = CUDA.zeros(Float32, 32, 32), CUDA.zeros(UInt32, 32, 32)
        () -> SFC.calculate_structure_function_batch!(s, c, SFT.L2SFType(), x, u, bins(0.3, 32); backend = DEV)
    end),
    "lag_sweep" => ("_gridded_lag_kernel", () -> begin
        u = randn(RNG, Float32, 2, 256, 256)
        sch = SFC.UniformLagSchedule((256, 256), (1.0, 1.0), (true, false))
        s, c = CUDA.zeros(Float32, 12), CUDA.zeros(Int, 12)
        () -> SFC.gridded_lag_sweep!(s, c, SFT.S3SFType(), reshape(u, 2, :), sch, collect(range(0.0f0, 12.0f0; length = 13)),
                                     Val(2); backend = DEV)
    end),
    "transform" => ("_spec_kernel|_lag_kernel", () -> begin
        u = randn(RNG, Float32, 2, 512, 512)
        sch = SFC.UniformLagSchedule((512, 512), (1.0, 1.0), (true, true))
        s, c = CUDA.zeros(Float32, 20), CUDA.zeros(Int, 20)
        ws = SFC.TransformWorkspace()
        () -> SFC.gridded_sweep!(s, c, SFT.L3SFType(), reshape(u, 2, :), sch, collect(range(0.0f0, 40.0f0; length = 21)),
                                 Val(2), SB.FastFourierTransformSpectralBackend(); backend = DEV, workspace = ws)
    end),
    "harmonic" => ("_harmonic_tile_kernel", () -> begin
        θ, φ = CUDA.CuArray(acos.(2 .* rand(RNG, 200_000) .- 1)), CUDA.CuArray(2π .* rand(RNG, 200_000))
        f = CUDA.CuArray(randn(RNG, ComplexF64, 200_000))
        () -> SFC.pseudo_coefficients_direct(f, θ, φ, 1, 128; backend = DEV)
    end),
    "sorted_line" => ("_line_sweep|_radix_", () -> begin
        x = CUDA.CuArray(reshape(rand(RNG, Float32, 1_000_000) .* 1000.0f0, 1, :))
        u = CUDA.CuArray(randn(RNG, Float32, 1, 1_000_000))
        () -> SFC.calculate_structure_function(SFT.L3SFType(), x, u, bins(2.0, 32), Int64, RAW; backend = DEV)
    end),
)

route = get(ARGS, 1, "")
haskey(ROUTES, route) || error("routes: $(join(sort(collect(keys(ROUTES))), ", ")); got \"$route\"")
pattern, make = ROUTES[route]
call = make()
CUDA.@sync call()
t = minimum(_ -> @elapsed(CUDA.@sync call()), 1:5)
Printf.@printf("route=%s kernel=%s wall=%.3f ms\n", route, pattern, 1e3 * t)
CUDA.@profile external = true CUDA.@sync call()
