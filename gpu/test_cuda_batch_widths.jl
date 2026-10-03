# Batch routing on real CUDA, across the axes the device *dispatches* on rather than the shapes a
# caller is likely to type: the coordinate and field widths, the bin type, and shared versus
# varying positions. Two recorded defects live on these axes — a fixed-position batch that read
# two components whatever the width, and the CUDA launcher's `Dv = D == 3 ? Val(3) : Val(2)`,
# whose `D` is the field width while the guard upstream reads the coordinate width.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using StructureFunctions.HelperFunctions: HelperFunctions as SFH
using ComputationalBackends: ComputationalBackends as CB
using Distances: Distances as DI

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts

failures = String[]

function compare(name, gs, gc, rs, rc; rtol = 1e-9)
    gs, gc = Array(gs), Array(gc)
    ds = maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps())
    dc = maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    @printf("%-56s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260918)
const N, T, NB = 500, 3, 8

# 1. Shared positions: the bin *type* picks the device route, so both spellings must agree at
#    both widths.
for D in (2, 3)
    x = rand(D, N)
    u = randn(D, N, T)
    ref_s = zeros(NB, T); ref_c = zeros(Int, NB, T)
    for t in 1:T
        r = SFC.calculate_structure_function(OP, x, u[:, :, t],
            collect(range(0.0, 1.0; length = NB + 1)), Float64, RAW;
            backend = SER)
        ref_s[:, t] .= r.sums; ref_c[:, t] .= r.counts
    end
    for (bname, bins) in (("LinearBinEdges", StructureFunctions.LinearBinEdges(
                               range(0.0, 1.0; length = NB + 1))),
                          ("raw vector", collect(range(0.0, 1.0; length = NB + 1))))
        g = SFC.gpu_calculate_structure_function_batch(OP, CUDA.CUDABackend(), x, u, bins, UInt32;
                                                       geometry = SFH.FlatGeometry{D}())
        compare("fixed-x batch D=$D $bname", reshape(collect(g.sums), NB, T),
                reshape(collect(g.counts), NB, T), ref_s, ref_c)
    end
end

# 2. A spherical batch: the coordinate width is two (lon, lat) while the field width is not, which
#    is the case the CUDA launcher's width rounding was never checked against.
let
    R = 6.371e6
    m = DI.Haversine(R)
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    db = collect(range(0.0, 9.0e6; length = NB + 1))
    for F in (2, 3)
        u = randn(F, N, T)
        r = SFC.calculate_structure_function(OP, x, u, db, RAW; backend = SER,
            distance_metric = m)
        g = SFC.calculate_structure_function(OP, x, u, db, RAW; backend = DEV,
            distance_metric = m)
        compare("spherical batch coords=2 field=$F", g.sums, g.counts, r.sums, r.counts)
    end
end

# 3. A histogram wider than the tiled kernel's shared-memory cap takes the global-atomic kernel,
#    which is a separate compile and only a device run establishes that it builds.
let
    x = rand(2, N)
    u = randn(2, N)
    for nb in (128, 129, 4000)
        bins = collect(range(0.0, 1.0; length = nb + 1))
        r = SFC.calculate_structure_function(OP, x, u, bins, Float64, RAW; backend = SER)
        g = SFC.calculate_structure_function(OP, x, u, bins, Float64, RAW; backend = DEV)
        compare("1D point histogram nb=$nb", g.sums, g.counts, r.sums, r.counts)
    end
    # and the same widths through the batch entry, whose staging differs
    u3 = randn(2, N, T)
    for nb in (128, 4000)
        bins = collect(range(0.0, 1.0; length = nb + 1))
        ref_s = zeros(nb, T); ref_c = zeros(Int, nb, T)
        for t in 1:T
            r = SFC.calculate_structure_function(OP, x, u3[:, :, t], bins, Float64, RAW;
                backend = SER)
            ref_s[:, t] .= r.sums; ref_c[:, t] .= r.counts
        end
        g = SFC.gpu_calculate_structure_function_batch(OP, CUDA.CUDABackend(), x, u3, bins, UInt32;
                                                       geometry = SFH.FlatGeometry{2}())
        compare("fixed-x batch histogram nb=$nb", reshape(collect(g.sums), nb, T),
                reshape(collect(g.counts), nb, T), ref_s, ref_c)
    end
end

if isempty(failures)
    println("\nCUDA BATCH WIDTHS OK")
else
    println("\nCUDA BATCH WIDTHS FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
