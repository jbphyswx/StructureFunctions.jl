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

# Widths 2 … 5 fit the native CUDA kernels' tiles in Float64; 6 fits the 2-D kernel's and not the
# 1-D kernel's; 7 fits neither, so the portable kernels take it.
for D in (2, 3, 5, 6, 7)
    x = rand(D, np); u = rand(D, np)
    r = SFC.calculate_structure_function(OP, x, u, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = SER, weights = w)
    g = SFC.calculate_structure_function(OP, x, u, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = DEV, weights = w)
    compare("1D point weighted D=$D", g.sums, g.counts, r.sums, r.counts)

    rj = SFC.calculate_structure_function(OP, x, u, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = SER, weights = w)
    gj = SFC.calculate_structure_function(OP, x, u, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = DEV, weights = w)
    compare("joint 2D point weighted D=$D", gj.sums, gj.counts, rj.sums, rj.counts)

    rs = SFC._dispatch_single_pass(SER, SFC.PointField{D}(), x, u, bins, Float64; weights = w)
    gs = SFC._dispatch_single_pass(DEV, SFC.PointField{D}(), x, u, bins, Float64; weights = w)
    compare("single-pass 1D point weighted D=$D", gs.sums, gs.counts, rs.sums, rs.counts)

    # auxiliary-axis batches, fixed and varying positions
    nt = 4
    ub = rand(D, np, nt)
    rb = SFC.calculate_structure_function(OP, x, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = SER, weights = w)
    gb = SFC.calculate_structure_function(OP, x, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = DEV, weights = w)
    compare("1D batch weighted (fixed x) D=$D", gb.sums, gb.counts, rb.sums, rb.counts)

    xb = rand(D, np, nt)
    rv = SFC.calculate_structure_function(OP, xb, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = SER, weights = w)
    gv = SFC.calculate_structure_function(OP, xb, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = DEV, weights = w)
    compare("1D batch weighted (varying x) D=$D", gv.sums, gv.counts, rv.sums, rv.counts)

    rjb = SFC.calculate_structure_function(OP, x, ub, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = SER, weights = w)
    gjb = SFC.calculate_structure_function(OP, x, ub, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = DEV, weights = w)
    compare("joint 2D batch weighted D=$D", gjb.sums, gjb.counts, rjb.sums, rjb.counts)

    rjv = SFC.calculate_structure_function(OP, xb, ub, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = SER, weights = w)
    gjv = SFC.calculate_structure_function(OP, xb, ub, bins, vb, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = DEV, weights = w)
    compare("joint 2D batch weighted (varying x) D=$D", gjv.sums, gjv.counts, rjv.sums, rjv.counts)

    nb, nv = length(bins) - 1, length(vb) - 1
    r2s = zeros(SFC.SINGLE_PASS_N, nb, nv); r2c = zeros(SFC.SINGLE_PASS_N, nb, nv)
    SFC.calculate_structure_functions_single_pass_2d!(r2s, r2c, x, u, bins, vb; backend = SER, weights = w)
    g2s = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nv); g2c = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nv)
    SFC.calculate_structure_functions_single_pass_2d!(g2s, g2c, x, u, bins, vb; backend = DEV, weights = w)
    compare("single-pass 2D point weighted D=$D", g2s, g2c, r2s, r2c)

    rps = zeros(SFC.SINGLE_PASS_N, nb, nt); rpc = zeros(SFC.SINGLE_PASS_N, nb, nt)
    SFC.calculate_structure_functions_single_pass_batch!(rps, rpc, x, ub, bins; backend = SER, weights = w)
    gps = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nt); gpc = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nt)
    SFC.calculate_structure_functions_single_pass_batch!(gps, gpc, x, ub, bins; backend = DEV, weights = w)
    compare("single-pass 1D batch weighted (fixed x) D=$D", gps, gpc, rps, rpc)

    rqs = zeros(SFC.SINGLE_PASS_N, nb, nv, nt); rqc = zeros(SFC.SINGLE_PASS_N, nb, nv, nt)
    SFC.calculate_structure_functions_single_pass_2d_batch!(rqs, rqc, xb, ub, bins, vb; backend = SER, weights = w)
    gqs = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nv, nt); gqc = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, nb, nv, nt)
    SFC.calculate_structure_functions_single_pass_2d_batch!(gqs, gqc, xb, ub, bins, vb; backend = DEV, weights = w)
    compare("single-pass 2D batch weighted (varying x) D=$D", gqs, gqc, rqs, rqc)
end

# Float32 input: the native kernels accumulate the pair mass in the count type's shared plane.
let D = 2, x = rand(Float32, 2, np), u = rand(Float32, 2, np), w32 = 0.3f0 .+ rand(Float32, np)
    b32, v32 = Float32.(bins), Float32.(vb)
    r = SFC.calculate_structure_function(OP, x, u, b32, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = SER, weights = w32)
    g = SFC.calculate_structure_function(OP, x, u, b32, Float64, SFO.StructureFunctionSumsAndCounts;
        backend = DEV, weights = w32)
    compare("1D point weighted Float32 D=$D", g.sums, g.counts, r.sums, r.counts; rtol = 1e-4)
    rj = SFC.calculate_structure_function(OP, x, u, b32, v32, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = SER, weights = w32)
    gj = SFC.calculate_structure_function(OP, x, u, b32, v32, Float64, SFO.StructureFunction2DSumsAndCounts;
        backend = DEV, weights = w32)
    compare("joint 2D point weighted Float32 D=$D", gj.sums, gj.counts, rj.sums, rj.counts; rtol = 1e-4)
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
        weights = w)
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
    r = SFC.calculate_structure_function(OP, x, u, bins, UInt32, SFO.StructureFunctionSumsAndCounts;
        backend = SER)
    g = SFC.calculate_structure_function(OP, x, u, bins, UInt32, SFO.StructureFunctionSumsAndCounts;
        backend = DEV)
    compare("1D point UNWEIGHTED D=$D", g.sums, g.counts, r.sums, r.counts)
end

if isempty(failures)
    println("\nCUDA PAIR WEIGHTS OK")
else
    println("\nCUDA PAIR WEIGHTS FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
