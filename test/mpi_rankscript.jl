# Launched under mpiexec by test_mpi.jl. Each rank computes its share via MPIBackend; rank 0
# compares the Allreduce'd result to a serial reference (identical seeded data on all ranks) and
# prints a marker the parent test greps for. Covers every entry family and shape, with both a
# serial and a threaded inner backend.
using ComputationalBackends: ComputationalBackends as CB
using MPI: MPI
MPI.Init()
using OhMyThreads: OhMyThreads
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StructureFunctions: MultiFields as MF, HarmonicNodes
using SpectralBackends: SpectralBackends as SB
using Random: Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sft = SFT.LongitudinalSecondOrderStructureFunctionType()

Random.seed!(123)                      # same data on every rank
N, B = 120, 4
x2 = rand(2, N); u2 = rand(2, N)
x3 = rand(3, N); u3 = rand(3, N)
ub = rand(2, N, B)                     # shared positions
xv = rand(2, N, B); uv = rand(2, N, B) # varying positions
bins = collect(range(0.0, 1.5, 21))
vbins = collect(range(-2.0, 2.0, 13))

# a periodic 8x8 grid for the gridded sweep
gsched = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
gdata = reshape(rand(2, 8, 8), 2, :)
gbins = collect(range(0.0, 0.5; length = 7))
gnb = length(gbins) - 1

# a sphere for the harmonic route, whose direct sum splits its point loop across ranks
hθ = acos.(clamp.(2 .* rand(N) .- 1, -1, 1))
hφ = 2π .* rand(N)
hx = permutedims(hcat(hφ, π / 2 .- hθ))
hu = randn(2, N)
hnodes = HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16)

# NaN marks an empty bin; two NaNs agree, a NaN opposite a number does not.
function same(a, b; rtol = 1e-8)
    av, bv = vec(collect(a)), vec(collect(b))
    length(av) == length(bv) || return false
    na, nb = isnan.(av), isnan.(bv)
    na == nb || return false
    keep = .!na
    return isapprox(av[keep], bv[keep]; rtol = rtol)
end

sc(o) = (o.sums, o.counts)
kw = (; verbose = false, show_progress = false)
raw = SFO.StructureFunctionSumsAndCounts

function cases(be)
    d = Dict{String, Any}()
    d["pf1d_D2"] = sc(SFC.calculate_structure_function(sft, x2, u2, bins; backend = be, output_type = raw, kw...))
    d["pf1d_D3"] = sc(SFC.calculate_structure_function(sft, x3, u3, bins; backend = be, output_type = raw, kw...))
    d["pf2d"] = sc(SFC.calculate_structure_function(sft, x2, u2, bins, vbins; backend = be, kw...))
    d["batch1d_fixed"] = sc(SFC.calculate_structure_function(sft, x2, ub, bins; backend = be, output_type = raw, kw...))
    d["batch1d_vary"] = sc(SFC.calculate_structure_function(sft, xv, uv, bins; backend = be, output_type = raw, kw...))
    d["batch2d_fixed"] = sc(SFC.calculate_structure_function(sft, x2, ub, bins, vbins; backend = be, kw...))
    sp1 = SFC.calculate_structure_functions_single_pass(x2, u2, bins; backend = be)
    d["sp1d"] = (sp1.S2.sums, sp1.L1T2.sums)
    sp2 = SFC.calculate_structure_functions_single_pass_2d(x2, u2, bins, vbins; backend = be)
    d["sp2d"] = (sp2.S2.sums, sp2.L1T2.sums)
    sp1b = SFC.calculate_structure_functions_single_pass(x2, ub, bins; backend = be)
    d["sp1d_batch"] = (sp1b.S2.sums, sp1b.L1T2.sums)
    sp2b = SFC.calculate_structure_functions_single_pass_2d(x2, ub, bins, vbins; backend = be)
    d["sp2d_batch"] = (sp2b.S2.sums, sp2b.L1T2.sums)
    return d
end

# Every remaining entry family, each isolated so one missing method reports rather than aborting.
const CASE_ERRORS = Dict{String, String}()

function extra_cases(be, tag)
    d = Dict{String, Any}()
    nb, nv = length(bins) - 1, length(vbins) - 1
    function add!(k, f)
        try
            d[k] = f()
        catch e
            d[k] = nothing
            CASE_ERRORS["$tag/$k"] = first(split(sprint(showerror, e), '\n'))
        end
    end

    add!("inplace_pf1d", function ()
        s, c = zeros(nb), zeros(UInt32, nb)
        SFC.calculate_structure_function!(s, c, sft, x2, u2, bins; backend = be, kw...)
        (s, c)
    end)
    add!("inplace_batch1d", function ()
        s, c = zeros(nb, B), zeros(UInt32, nb, B)
        SFC.calculate_structure_function!(s, c, sft, x2, ub, bins; backend = be, kw...)
        (s, c)
    end)
    add!("inplace_pf2d", function ()
        s, c = zeros(nb, nv), zeros(UInt32, nb, nv)
        SFC.calculate_structure_function!(s, c, sft, x2, u2, bins, vbins; backend = be, kw...)
        (s, c)
    end)
    add!("inplace_batch2d", function ()
        s, c = zeros(nb, nv, B), zeros(UInt32, nb, nv, B)
        SFC.calculate_structure_function!(s, c, sft, x2, ub, bins, vbins; backend = be, kw...)
        (s, c)
    end)
    add!("drv_batch1d", function ()
        s, c = zeros(nb, B), zeros(UInt32, nb, B)
        SFC.calculate_structure_function_batch!(s, c, sft, x2, ub, bins; backend = be)
        (s, c)
    end)
    add!("drv_batch2d", function ()
        s, c = zeros(nb, nv, B), zeros(UInt32, nb, nv, B)
        SFC.calculate_structure_function_2d_batch!(s, c, sft, x2, ub, bins, vbins; backend = be)
        (s, c)
    end)
    add!("drv_sp1d", function ()
        s, c = zeros(SFC.SINGLE_PASS_N, nb, B), zeros(UInt32, SFC.SINGLE_PASS_N, nb, B)
        SFC.calculate_structure_functions_single_pass_batch!(s, c, x2, ub, bins; backend = be)
        (s, c)
    end)
    add!("drv_sp2d", function ()
        s = zeros(SFC.SINGLE_PASS_N, nb, nv, B)
        c = zeros(UInt32, SFC.SINGLE_PASS_N, nb, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x2, ub, bins, vbins; backend = be)
        (s, c)
    end)
    add!("inplace_sp1d", function ()
        s, c = zeros(SFC.SINGLE_PASS_N, nb), zeros(UInt32, SFC.SINGLE_PASS_N, nb)
        SFC.calculate_structure_functions_single_pass!(s, c, x2, u2, bins; backend = be)
        (s, c)
    end)
    add!("inplace_sp2d", function ()
        s, c = zeros(SFC.SINGLE_PASS_N, nb, nv), zeros(UInt32, SFC.SINGLE_PASS_N, nb, nv)
        SFC.calculate_structure_functions_single_pass_2d!(s, c, x2, u2, bins, vbins; backend = be)
        (s, c)
    end)
    add!("tensor2", function ()
        t = SFC.calculate_structure_function_tensor(Val(2), x2, u2, bins; backend = be,
            output_type = SFO.StructureFunctionTensorSumsAndCounts)
        (t.sums, t.counts)
    end)
    add!("multifield", function ()
        s, c = zeros(nb), zeros(UInt32, nb)
        f = MF.Fields{2, 1, 0, typeof(u2)}(u2)
        SFC.calculate_structure_function!(s, c, sft, x2, f, bins; backend = be)
        (s, c)
    end)
    add!("harmonic", function ()
        r = SFC.calculate_structure_function(sft, hx, hu, hnodes, SB.DirectSumSpectralBackend();
            backend = be, verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
        (r.sums, r.counts)
    end)
    add!("gridded_sweep", function ()
        s, c = zeros(gnb), zeros(Int, gnb)
        SFC.gridded_lag_sweep!(s, c, sft, gdata, gsched, gbins, Val(2), Val(1), Val(0); backend = be)
        (s, c)
    end)
    return d
end

inners = (("serial", CB.SerialBackend()), ("threaded", CB.ThreadedBackend()))
results = Dict(
    name => merge(cases(CB.MPIBackend(inner)), extra_cases(CB.MPIBackend(inner), name))
    for (name, inner) in inners
)

status = 0
if rank == 0
    ref = merge(cases(CB.SerialBackend()), extra_cases(CB.SerialBackend(), "serial_ref"))
    failures = String[]
    for (iname, got) in results, k in sort!(collect(keys(ref)))
        r, g = ref[k], got[k]
        if r === nothing || g === nothing
            why = get(CASE_ERRORS, "$iname/$k", get(CASE_ERRORS, "serial_ref/$k", "no result"))
            push!(failures, "$iname/$k [$why]")
        elseif !all(p -> same(p[1], p[2]), zip(r, g))
            push!(failures, "$iname/$k")
        end
    end
    if isempty(failures)
        println("MPI parity OK: np=$(MPI.Comm_size(comm)) nthreads=$(Threads.nthreads()) cases=$(length(ref) * length(inners))")
    else
        println(stderr, "MPI parity FAILED for: ", join(failures, ", "))
        status = 1
    end
end
MPI.Finalize()
exit(status)