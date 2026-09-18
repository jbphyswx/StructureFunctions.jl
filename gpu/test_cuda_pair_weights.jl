# Pair weights on the device, on real CUDA: every weighted kernel must compile (which `KA.CPU()`
# cannot establish) and agree with the serial weighted answer.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using StructureFunctions.MultiFields: MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.S2SFType()

failures = String[]

function compare(name, got_s, got_c, ref_s, ref_c; rtol = 1e-9)
    gs, gc = Array(got_s), Array(got_c)
    rs, rc = Array(ref_s), Array(ref_c)
    scale = maximum(abs, rs) + eps()
    ds = maximum(abs.(gs .- rs)) / scale
    dc = maximum(abs.(gc .- rc)) / (maximum(abs, rc) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    @printf("%-46s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260918)
np = 4000
bins = collect(range(0.0, 1.0; length = 9)) .+ 0.0137   # off-lattice, see gap AP
vb = collect(range(0.0, 2.0; length = 5)) .+ 0.011
w = 0.3 .+ rand(np)

for D in (2, 3)
    x = rand(D, np); u = rand(D, np)
    r = SFC.calculate_structure_function(OP, x, u, bins, Float64; backend = SER, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    g = SFC.calculate_structure_function(OP, x, u, bins, Float64; backend = DEV, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    compare("1D point weighted D=$D", g.sums, g.counts, r.sums, r.counts)

    rj = SFC.calculate_structure_function(OP, x, u, bins, vb, Float64; backend = SER, weights = w,
        verbose = false, output_type = SFO.StructureFunction2DSumsAndCounts)
    gj = SFC.calculate_structure_function(OP, x, u, bins, vb, Float64; backend = DEV, weights = w,
        verbose = false, output_type = SFO.StructureFunction2DSumsAndCounts)
    compare("joint 2D point weighted D=$D", gj.sums, gj.counts, rj.sums, rj.counts)

    rs = SFC._dispatch_single_pass(SER, SFC.PointField{D}(), x, u, bins;
        count_eltype = Float64, weights = w)
    gs = SFC._dispatch_single_pass(DEV, SFC.PointField{D}(), x, u, bins;
        count_eltype = Float64, weights = w, verbose = false)
    compare("single-pass 1D point weighted D=$D", gs.sums, gs.counts, rs.sums, rs.counts)

    # auxiliary-axis batches, fixed and varying positions
    nt = 4
    ub = rand(D, np, nt)
    rb = SFC.calculate_structure_function(OP, x, ub, bins, Float64; backend = SER, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    gb = SFC.calculate_structure_function(OP, x, ub, bins, Float64; backend = DEV, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    compare("1D batch weighted (fixed x) D=$D", gb.sums, gb.counts, rb.sums, rb.counts)

    xb = rand(D, np, nt)
    rv = SFC.calculate_structure_function(OP, xb, ub, bins, Float64; backend = SER, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    gv = SFC.calculate_structure_function(OP, xb, ub, bins, Float64; backend = DEV, weights = w,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    compare("1D batch weighted (varying x) D=$D", gv.sums, gv.counts, rv.sums, rv.counts)

    rjb = SFC.calculate_structure_function(OP, x, ub, bins, vb, Float64; backend = SER, weights = w,
        verbose = false, output_type = SFO.StructureFunction2DSumsAndCounts)
    gjb = SFC.calculate_structure_function(OP, x, ub, bins, vb, Float64; backend = DEV, weights = w,
        verbose = false, output_type = SFO.StructureFunction2DSumsAndCounts)
    compare("joint 2D batch weighted D=$D", gjb.sums, gjb.counts, rjb.sums, rjb.counts)
end

# multi-field
let D = 2
    x = rand(D, np)
    fields = MF.Fields(vectors = (rand(D, np),), scalars = (rand(np),))
    mixed = SFT.MixedSFType{1, 0, 2}()
    nb = length(bins) - 1
    rs = zeros(Float64, nb); rc = zeros(Float64, nb)
    SFC.serial_calculate_structure_function!(rs, rc, mixed, x, fields, bins; weights = w)
    gs = zeros(Float64, nb); gc = zeros(Float64, nb)
    SFC.gpu_calculate_structure_function_fields!(DEV, gs, gc, mixed, x, fields, bins;
        weights = w, verbose = false)
    compare("multi-field weighted", gs, gc, rs, rc)
end

# The moment tensor: the device kernel gained the weight argument its siblings carry, so this is
# the first CUDA compile of that signature.
for (D, P) in ((2, 2), (3, 2), (2, 3))
    x = rand(D, np); u = rand(D, np)
    nb = length(bins) - 1
    rs = zeros(Float64, ntuple(_ -> D, P)..., nb); rc = zeros(Float64, nb)
    SFC.calculate_structure_function_tensor!(rs, rc, Val(P), x, u, bins; backend = SER, weights = w)
    gs = zeros(Float64, ntuple(_ -> D, P)..., nb); gc = zeros(Float64, nb)
    SFC.calculate_structure_function_tensor!(gs, gc, Val(P), x, u, bins; backend = DEV, weights = w)
    compare("moment tensor weighted D=$D P=$P", gs, gc, rs, rc)
end

# A constant weight k scales every sum and count by exactly k², whatever the operator or binning,
# so this fails for a kernel that takes the weight and does not apply it.
let D = 2, P = 2, k = 3.0
    x = rand(D, np); u = rand(D, np)
    nb = length(bins) - 1
    us = zeros(Float64, D, D, nb); uc = zeros(Float64, nb)
    SFC.calculate_structure_function_tensor!(us, uc, Val(P), x, u, bins; backend = DEV)
    ks = zeros(Float64, D, D, nb); kc = zeros(Float64, nb)
    SFC.calculate_structure_function_tensor!(ks, kc, Val(P), x, u, bins; backend = DEV,
        weights = fill(k, np))
    compare("moment tensor k^2 scaling on device", ks, kc, k^2 .* us, k^2 .* uc)
end

# the unweighted device answer must be untouched by all of this
for D in (2, 3)
    x = rand(D, np); u = rand(D, np)
    r = SFC.calculate_structure_function(OP, x, u, bins, UInt32; backend = SER,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    g = SFC.calculate_structure_function(OP, x, u, bins, UInt32; backend = DEV,
        verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    compare("1D point UNWEIGHTED D=$D", g.sums, g.counts, r.sums, r.counts)
end

if isempty(failures)
    println("\nCUDA PAIR WEIGHTS OK")
else
    println("\nCUDA PAIR WEIGHTS FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
