# Launched under mpiexec by test_mpi.jl. Each rank computes its share via MPIBackend and checks the Allreduce'd results of
# its rows against a serial reference (identical seeded data on all ranks); one row per mechanism of the MPI extension:
# each partial family's reduction, the batch executor plain and culled, the line's split sweep, the harmonic sum, and
# the in-place forms.
using ComputationalBackends: ComputationalBackends as CB
using MPI: MPI
MPI.Init()
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StructureFunctions: MultiFields as MF, HarmonicNodes
using SpectralBackends: SpectralBackends as SB
using Random: Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sft = SFT.L2SFType()
RAW = SFO.StructureFunctionSumsAndCounts

Random.seed!(123)                      # same data on every rank
N, B = 60, 3
x2 = rand(2, N); u2 = randn(2, N)
x1 = rand(1, N); u1 = randn(1, N)      # points on a line, swept in sorted order
ub = randn(2, N, B)                    # shared positions
θ = randn(N)
bins = collect(range(0.0, 1.5, 21))
vbins = collect(range(-3.0, 3.0; length = 7))
abins = collect(range(prevfloat(0.0), π; length = 5))
angle = SFC.SeparationAngleAxis([1.0, 0.0])
w = rand(N) .+ 0.5                     # pair weights, one per point
nb, nv, na, NI = length(bins) - 1, length(vbins) - 1, length(abins) - 1, SFC.SINGLE_PASS_N
always = SFC.AlwaysCulling()

# a sphere for the harmonic route, whose direct sum splits its point loop across ranks
hθ = acos.(clamp.(2 .* rand(N) .- 1, -1, 1))
hφ = 2π .* rand(N)
hx = permutedims(hcat(hφ, π / 2 .- hθ))
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
stacked(r) = (stack([r[k].sums for k in keys(r) if k !== :helmholtz]),
              stack([r[k].counts for k in keys(r) if k !== :helmholtz]))

"""`f!(s, c)` run once into the zeroed buffers `s`, `c`, returned."""
into(f!, s, c) = (f!(s, c); (s, c))

const ROWS = (
    ("point_culled_weighted_inplace", be -> into(zeros(nb), zeros(nb)) do s, c
        SFC.calculate_structure_function!(s, c, sft, x2, u2, bins; backend = be, culling = always, weights = w)
    end),
    ("line", be -> sc(SFC.calculate_structure_function(sft, x1, u1, bins, RAW; backend = be))),
    ("joint", be -> sc(SFC.calculate_structure_function(sft, x2, u2, bins, vbins; backend = be))),
    ("batch1d", be -> sc(SFC.calculate_structure_function(sft, x2, ub, bins, RAW; backend = be))),
    ("batch2d_angle_culled_driver", be -> into(zeros(nb, na, B), zeros(UInt32, nb, na, B)) do s, c
        SFC.calculate_structure_function_2d_batch!(s, c, sft, x2, ub, bins, abins; backend = be, culling = always,
                                                   second_axis = angle)
    end),
    ("sp1d_inplace", be -> into(zeros(NI, nb), zeros(UInt32, NI, nb)) do s, c
        SFC.calculate_structure_functions_single_pass!(s, c, x2, u2, bins; backend = be)
    end),
    ("sp2d", be -> stacked(SFC.calculate_structure_functions_single_pass_2d(x2, u2, bins, vbins; backend = be))),
    ("tensor", be -> sc(SFC.calculate_structure_function_tensor(Val(2), x2, u2, bins,
                                                                SFO.StructureFunctionTensorSumsAndCounts; backend = be))),
    ("multifield", be -> into(zeros(nb), zeros(UInt32, nb)) do s, c
        SFC.calculate_structure_function!(s, c, SFT.MixedSFType{1, 0, 2}(), x2, MF.Fields(vectors = (u2,), scalars = (θ,)),
                                          bins; backend = be)
    end),
    ("harmonic", be -> sc(SFC.calculate_structure_function(sft, hx, u2, hnodes, SB.DirectSumSpectralBackend(), RAW;
                                                           backend = be))),
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

results = [run_row(name, f, CB.MPIBackend(CB.SerialBackend())) for (name, f) in ROWS]

# Each rank checks the rows of every `nranks`-th family (a row name less its entry-form suffix).
nranks = MPI.Comm_size(comm)
family(name) = replace(name, r"_(inplace|driver)$" => "")
families = unique(family(name) for (name, _) in ROWS)
failures = String[]
for ((name, f), g) in zip(ROWS, results)
    (findfirst(==(family(name)), families) - 1) % nranks == rank || continue
    r = run_row("$name/serial_ref", f, CB.SerialBackend())
    if r === nothing || g === nothing
        why = get(CASE_ERRORS, name, get(CASE_ERRORS, "$name/serial_ref", "no result"))
        push!(failures, "$name [$why]")
    elseif !all(p -> same(p[1], p[2]), zip(r, g))
        push!(failures, name)
    end
end
isempty(failures) || println(stderr, "MPI parity FAILED on rank $rank for: ", join(failures, ", "))
n_failed = MPI.Allreduce(length(failures), +, comm)
n_failed == 0 && rank == 0 && println("MPI parity OK: np=$nranks cases=$(length(results))")
status = n_failed == 0 ? 0 : 1
MPI.Finalize()
exit(status)
