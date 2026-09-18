# The separation-angle second axis on real CUDA: the joint kernels gained an axis argument, and
# only a device compile can establish that the generated tiled family and the three global-atomic
# kernels still build. `KA.CPU()` accepts host values a device rejects.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()

failures = String[]

function compare(name, got, ref; rtol = 1e-9)
    gs, gc = Array(got.sums), Array(got.counts)
    rs, rc = Array(ref.sums), Array(ref.counts)
    ds = maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps())
    dc = maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    @printf("%-52s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260918)
const NP = 4000
const DBINS = collect(range(0.0, 1.0; length = 9)) .+ 0.0137   # off-lattice, see gap AP
const SRC = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))

# A histogram under the shared-memory cap takes the tiled kernel; one over it takes the
# global-atomic kernels, which are a different code path.
for (route, n_angle) in (("tiled", 5), ("global atomic", 401))
    abins = collect(range(prevfloat(0.0), π; length = n_angle))
    for D in (2, 3)
        x, u = rand(D, NP), rand(D, NP)
        src = SFC.SeparationAngleAxis(D == 2 ? SA.SVector(1.0, 0.0) : SA.SVector(1.0, 0.0, 0.0))
        ref = SFC.calculate_structure_function(OP, x, u, DBINS, abins; backend = SER,
            verbose = false, second_axis = src)
        got = SFC.calculate_structure_function(OP, x, u, DBINS, abins; backend = DEV,
            verbose = false, second_axis = src)
        compare("angle axis $route D=$D", got, ref)
    end
end

# The value axis must be untouched by the extra argument, on both histogram routes.
for (route, n_val) in (("tiled", 5), ("global atomic", 401))
    vbins = collect(range(0.0, 2.0; length = n_val)) .+ 0.011
    x, u = rand(2, NP), rand(2, NP)
    ref = SFC.calculate_structure_function(OP, x, u, DBINS, vbins; backend = SER, verbose = false)
    got = SFC.calculate_structure_function(OP, x, u, DBINS, vbins; backend = DEV, verbose = false)
    compare("value axis UNCHANGED $route", got, ref)
end

# The angle is read off `X2 - X1`, which is the separation only on a flat metric.
let x = rand(2, 64), u = rand(2, 64)
    abins = collect(range(prevfloat(0.0), π; length = 5))
    refused = try
        SFC.calculate_structure_function(OP, x, u, DBINS, abins; backend = DEV, verbose = false,
            second_axis = SRC, distance_metric = SFC.DI.SphericalAngle())
        false
    catch e
        e isa ArgumentError
    end
    refused || push!(failures, "curved metric refusal")
    @printf("%-52s %s\n", "angle axis on a curved metric is refused by name",
            refused ? "ok" : "FAILED")
end

if isempty(failures)
    println("\nCUDA SECOND AXIS OK")
else
    println("\nCUDA SECOND AXIS FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
