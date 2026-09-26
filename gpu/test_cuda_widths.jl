# Coordinate width on real CUDA. A width outside the set the package compiles ahead of time is one
# more kernel instantiation, so every route must compile it on first launch and give the serial
# answer. `KA.CPU()` cannot establish this: it compiles no kernels, and the staged-point loaders
# read a `@localmem` tile through an `ntuple` closure, which only a device compile exercises.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts
# In Float64 the native CUDA kernels' tiles fit through width 5; the 2-D kernel's also at 6.
const WIDTHS = (2, 3, 4, 5, 6, 7)

failures = String[]

function compare(name, gs, gc, rs, rc; rtol = 1e-9)
    gs, gc = Array(gs), Array(gc)
    ds = maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps())
    dc = maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    @printf("%-52s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
const N, T, NB, NV = 400, 3, 8, 5
const BINS = collect(range(0.0, 1.0; length = NB + 1))
const VBINS = collect(range(-3.0, 3.0; length = NV + 1))
const ABINS = collect(range(prevfloat(0.0), Float64(π); length = 5))

for D in WIDTHS
    Random.seed!(4400 + D)
    x, u = rand(D, N), randn(D, N)
    xb, ub = rand(D, N, T), randn(D, N, T)
    ax = SFC.SeparationAngleAxis(SA.SVector(ntuple(d -> d == 1 ? 1.0 : 0.0, D)))

    r = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW;
        backend = SER)
    g = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW;
        backend = DEV)
    compare("point 1D D=$D", g.sums, g.counts, r.sums, r.counts)

    rj = SFC.calculate_structure_function(OP, x, u, BINS, VBINS; backend = SER)
    gj = SFC.calculate_structure_function(OP, x, u, BINS, VBINS; backend = DEV)
    compare("joint value D=$D", gj.sums, gj.counts, rj.sums, rj.counts)

    ra = SFC.calculate_structure_function(OP, x, u, BINS, ABINS;
        backend = SER, second_axis = ax)
    ga = SFC.calculate_structure_function(OP, x, u, BINS, ABINS;
        backend = DEV, second_axis = ax)
    compare("joint angle D=$D", ga.sums, ga.counts, ra.sums, ra.counts)
    # the control: the angle histogram is not the value histogram, so a dropped keyword cannot pass
    rv = SFC.calculate_structure_function(OP, x, u, BINS, ABINS; backend = SER)
    if collect(ra.counts) == collect(rv.counts)
        push!(failures, "angle control D=$D")
        println("angle control D=$D                                   FAILED (angle == value)")
    end

    rs = zeros(SFC.SINGLE_PASS_N, NB); rc = zeros(Int, SFC.SINGLE_PASS_N, NB)
    SFC.calculate_structure_functions_single_pass!(rs, rc, x, u, BINS; backend = SER)
    gs = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, NB); gc = CUDA.zeros(Int, SFC.SINGLE_PASS_N, NB)
    SFC.calculate_structure_functions_single_pass!(gs, gc, x, u, BINS; backend = DEV)
    compare("single-pass 1D D=$D", gs, gc, rs, rc)

    r2s = zeros(SFC.SINGLE_PASS_N, NB, NV); r2c = zeros(Int, SFC.SINGLE_PASS_N, NB, NV)
    SFC.calculate_structure_functions_single_pass_2d!(r2s, r2c, x, u, BINS, VBINS; backend = SER)
    g2s = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, NB, NV); g2c = CUDA.zeros(Int, SFC.SINGLE_PASS_N, NB, NV)
    SFC.calculate_structure_functions_single_pass_2d!(g2s, g2c, x, u, BINS, VBINS; backend = DEV)
    compare("single-pass 2D D=$D", g2s, g2c, r2s, r2c)

    bs = zeros(NB, T); bc = zeros(Int, NB, T)
    SFC.calculate_structure_function_batch!(bs, bc, OP, xb, ub, BINS; backend = SER)
    gbs = CUDA.zeros(Float64, NB, T); gbc = CUDA.zeros(Int, NB, T)
    SFC.calculate_structure_function_batch!(gbs, gbc, OP, xb, ub, BINS; backend = DEV)
    compare("slice batch 1D D=$D", gbs, gbc, bs, bc)

    js = zeros(NB, NV, T); jc = zeros(Int, NB, NV, T)
    SFC.calculate_structure_function_2d_batch!(js, jc, OP, xb, ub, BINS, VBINS; backend = SER)
    gjs = CUDA.zeros(Float64, NB, NV, T); gjc = CUDA.zeros(Int, NB, NV, T)
    SFC.calculate_structure_function_2d_batch!(gjs, gjc, OP, xb, ub, BINS, VBINS; backend = DEV)
    compare("slice batch joint D=$D", gjs, gjc, js, jc)

    ps = zeros(SFC.SINGLE_PASS_N, NB, T); pc = zeros(Int, SFC.SINGLE_PASS_N, NB, T)
    SFC.calculate_structure_functions_single_pass_batch!(ps, pc, xb, ub, BINS; backend = SER)
    gps = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, NB, T); gpc = CUDA.zeros(Int, SFC.SINGLE_PASS_N, NB, T)
    SFC.calculate_structure_functions_single_pass_batch!(gps, gpc, xb, ub, BINS; backend = DEV)
    compare("slice batch sp1d D=$D", gps, gpc, ps, pc)

    qs = zeros(SFC.SINGLE_PASS_N, NB, NV, T); qc = zeros(Int, SFC.SINGLE_PASS_N, NB, NV, T)
    SFC.calculate_structure_functions_single_pass_2d_batch!(qs, qc, xb, ub, BINS, VBINS;
        backend = SER)
    gqs = CUDA.zeros(Float64, SFC.SINGLE_PASS_N, NB, NV, T); gqc = CUDA.zeros(Int, SFC.SINGLE_PASS_N, NB, NV, T)
    SFC.calculate_structure_functions_single_pass_2d_batch!(gqs, gqc, xb, ub, BINS, VBINS;
        backend = DEV)
    compare("slice batch sp2d D=$D", gqs, gqc, qs, qc)
end

# Above the 2D tiled kernels' staging budget there is no sibling to widen into, so the refusal is
# derived from the budget and must name it.
let D = 64
    xb, ub = rand(D, 32, 2), randn(D, 32, 2)
    s = CUDA.zeros(Float64, NB, 2); c = CUDA.zeros(Int, NB, 2)
    threw = false
    try
        SFC.calculate_structure_function_2d_batch!(CUDA.zeros(Float64, NB, NV, 2), CUDA.zeros(Int, NB, NV, 2), OP,
            xb, ub, BINS, VBINS; backend = DEV)
    catch e
        budget = SFC.gpu_static_smem_budget(SFC.gpu_device_caps(CUDA.CUDABackend()))
        threw = e isa ArgumentError && occursin("$budget", sprint(showerror, e))
    end
    threw || push!(failures, "2D batch budget refusal at D=$D")
    println(threw ? "2D batch refuses D=$D by its staging budget           ok" :
                    "2D batch refuses D=$D by its staging budget           FAILED")
    # the 1D tier stages nothing above its budget, so the same width runs there
    SFC.calculate_structure_function_batch!(s, c, OP, xb, ub, BINS; backend = DEV)
    rs = zeros(NB, 2); rc = zeros(Int, NB, 2)
    SFC.calculate_structure_function_batch!(rs, rc, OP, xb, ub, BINS; backend = SER)
    compare("slice batch 1D D=$D (wide kernel)", s, c, rs, rc)
end

println()
if isempty(failures)
    println("CUDA WIDTHS OK")
    exit(0)
end
println("CUDA WIDTHS FAILED: ", join(failures, ", "))
exit(1)
