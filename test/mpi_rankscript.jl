# Launched under mpiexec by test_mpi.jl. Each rank computes its share via MPIBackend; rank 0
# compares the Allreduce'd result to a serial reference (identical seeded data on all ranks) and
# prints a marker the parent test greps for. Covers every entry family and shape, with both a
# serial and a threaded inner backend.
using ComputationalBackends: ComputationalBackends as CB
using MPI: MPI
MPI.Init()
using OhMyThreads: OhMyThreads
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
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
    d["sp1d"] = (sp1.S2.values, sp1.L1T2.values)
    sp2 = SFC.calculate_structure_functions_single_pass_2d(x2, u2, bins, vbins; backend = be)
    d["sp2d"] = (sp2.S2.sums, sp2.L1T2.sums)
    sp1b = SFC.calculate_structure_functions_single_pass(x2, ub, bins; backend = be)
    d["sp1d_batch"] = (sp1b.S2.values, sp1b.L1T2.values)
    sp2b = SFC.calculate_structure_functions_single_pass_2d(x2, ub, bins, vbins; backend = be)
    d["sp2d_batch"] = (sp2b.S2.sums, sp2b.L1T2.sums)
    return d
end

inners = (("serial", CB.SerialBackend()), ("threaded", CB.ThreadedBackend()))
results = Dict(name => cases(CB.MPIBackend(inner)) for (name, inner) in inners)

status = 0
if rank == 0
    ref = cases(CB.SerialBackend())
    failures = String[]
    for (iname, got) in results, k in sort!(collect(keys(ref)))
        all(p -> same(p[1], p[2]), zip(ref[k], got[k])) || push!(failures, "$iname/$k")
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