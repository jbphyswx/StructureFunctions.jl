# Launched under mpiexec by test_mpi.jl. Each rank computes its share via MPIBackend and checks the Allreduce'd results of
# its rows against a serial reference (identical seeded data on all ranks); one row per reduction path of the MPI extension.
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
N, B = 60, 3
x2 = rand(2, N); u2 = rand(2, N)
x4 = rand(4, N); u4 = rand(4, N)       # a width outside the set the SIMD kernels specialize for
ub = rand(2, N, B)                     # shared positions
bins = collect(range(0.0, 1.5, 21))
abins = collect(range(prevfloat(0.0), π; length = 5))
angle = SFC.SeparationAngleAxis([1.0, 0.0])
w = rand(N) .+ 0.5                     # pair weights, one per point
nb, na = length(bins) - 1, length(abins) - 1
always = SFC.AlwaysCulling()

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
const SER, THR = CB.SerialBackend(), CB.ThreadedBackend()

# (name, inner backend, computation): one row per way the MPI extension splits and reduces a sweep.
const ROWS = (
    ("pf1d_D4_culled_weighted_inplace", SER, be -> begin
        s, c = zeros(nb), zeros(nb)
        SFC.calculate_structure_function!(s, c, sft, x4, u4, bins; backend = be, culling = always, weights = w)
        (s, c)
    end),
    ("multifield", THR, be -> begin
        s, c = zeros(nb), zeros(UInt32, nb)
        SFC.calculate_structure_function!(s, c, sft, x2, MF.Fields{2, 1, 0, typeof(u2)}(u2), bins; backend = be)
        (s, c)
    end),
    ("batch2d_angle_culled_inplace", SER, be -> begin
        s, c = zeros(nb, na, B), zeros(UInt32, nb, na, B)
        SFC.calculate_structure_function!(s, c, sft, x2, ub, bins, abins; backend = be, culling = always,
            second_axis = angle)
        (s, c)
    end),
    ("batch2d_angle_culled_driver", SER, be -> begin
        s, c = zeros(nb, na, B), zeros(UInt32, nb, na, B)
        SFC.calculate_structure_function_2d_batch!(s, c, sft, x2, ub, bins, abins; backend = be, culling = always,
            second_axis = angle)
        (s, c)
    end),
    ("harmonic", SER, be -> sc(SFC.calculate_structure_function(sft, hx, hu, hnodes, SB.DirectSumSpectralBackend(),
        SFO.StructureFunctionSumsAndCounts; backend = be))),
    ("gridded_sweep", SER, be -> begin
        s, c = zeros(gnb), zeros(Int, gnb)
        SFC.gridded_lag_sweep!(s, c, sft, gdata, gsched, gbins, Val(2), Val(1), Val(0); backend = be)
        (s, c)
    end),
)

# Each row isolated, so one missing method reports rather than aborting.
const CASE_ERRORS = Dict{String, String}()

function run_row(key, f, be)
    try
        return f(be)
    catch e
        CASE_ERRORS[key] = first(split(sprint(showerror, e), '\n'))
        return nothing
    end
end

results = [run_row("$name/$(nameof(typeof(inner)))", f, CB.MPIBackend(inner)) for (name, inner, f) in ROWS]

# Each rank checks the rows of every `nranks`-th family (a row name less its entry-form suffix).
nranks = MPI.Comm_size(comm)
family(name) = replace(name, r"_(inplace|driver)$" => "")
families = unique(family(name) for (name, _, _) in ROWS)
failures = String[]
for ((name, inner, f), g) in zip(ROWS, results)
    (findfirst(==(family(name)), families) - 1) % nranks == rank || continue
    r = run_row("$name/serial_ref", f, CB.SerialBackend())
    iname = nameof(typeof(inner))
    if r === nothing || g === nothing
        why = get(CASE_ERRORS, "$name/$iname", get(CASE_ERRORS, "$name/serial_ref", "no result"))
        push!(failures, "$iname/$name [$why]")
    elseif !all(p -> same(p[1], p[2]), zip(r, g))
        push!(failures, "$iname/$name")
    end
end
isempty(failures) || println(stderr, "MPI parity FAILED on rank $rank for: ", join(failures, ", "))
n_failed = MPI.Allreduce(length(failures), +, comm)
n_failed == 0 && rank == 0 &&
    println("MPI parity OK: np=$nranks nthreads=$(Threads.nthreads()) cases=$(length(results))")
status = n_failed == 0 ? 0 : 1
MPI.Finalize()
exit(status)
