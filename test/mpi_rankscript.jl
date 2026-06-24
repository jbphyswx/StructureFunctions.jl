# Launched under mpiexec by test_mpi.jl. Each rank computes its share via MPIBackend;
# rank 0 compares the Allreduce'd result to a serial reference (identical seeded data on all
# ranks) and prints a marker the parent test greps for.
using MPI
MPI.Init()
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sft = SFT.LongitudinalSecondOrderStructureFunctionType()
Random.seed!(123)                      # same data on every rank
N = 200
x = rand(3, N); u = rand(3, N)
bins = collect(range(0.0, 1.5, 21))

res = SFC.calculate_structure_function(
    sft, x, u, bins; backend = SFC.MPIBackend(),
    verbose = false, show_progress = false, return_sums_and_counts = true,
)

if rank == 0
    ref = SFC.calculate_structure_function(
        sft, x, u, bins; backend = SFC.SerialBackend(),
        verbose = false, show_progress = false, return_sums_and_counts = true,
    )
    ok = ref.counts == res.counts && isapprox(ref.sums, res.sums; rtol = 1e-8)
    println(ok ? "MPI_PARITY_OK np=$(MPI.Comm_size(comm))" : "MPI_PARITY_FAIL")
end
MPI.Finalize()
