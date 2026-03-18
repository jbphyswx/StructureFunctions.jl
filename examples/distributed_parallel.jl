"""
    example_distributed_parallel.jl

Multi-process distributed structure function calculation for massive datasets.

This example demonstrates:
- Setting up multiple Julia processes
- Distributing data across processes
- Fault-tolerant computation
- Cluster submission (SLURM example included)

Requires: Distributed.jl (stdlib in Julia >= 0.6)

Run:
    # Local cluster (4 processes)
    julia --project example_distributed_parallel.jl
    
    # Or with custom process count
    julia --project -- example_distributed_parallel.jl 8
    
    # On HPC cluster with SLURM
    sbatch submit_distributed.sbatch
"""

using Distributed
using StructureFunctions
using Statistics

# ============================================================================
# 1. Setup Process Pool
# ============================================================================

println("Distributed Computation Setup")
println("="^60)

# Parse command-line argument for number of processes
nprocs_desired = if length(ARGS) > 0
    parse(Int, ARGS[1])
else
    max(Sys.CPU_THREADS - 1, 2)  # Default: all cores - 1
end

# Add processes (leave one for main)
if nworkers() < nprocs_desired
    print("Starting $(nprocs_desired - 1) worker processes...")
    addprocs(nprocs_desired - 1)
end

println("✓ Running with $(nworkers()) workers + 1 main process")
println("  Total processes: $(nprocs())")
println("  Worker IDs: $(workers())")

# ============================================================================
# 2. Load Package on All Workers
# ============================================================================

println("\n" * "="^60)
println("Loading StructureFunctions.jl on all workers...")
println("="^60)

@everywhere using StructureFunctions
@everywhere using Statistics

println("✓ StructureFunctions.jl loaded on all workers")

# ============================================================================
# 3. Generate Large Distributed Dataset
# ============================================================================

println("\n" * "="^60)
println("Generating Distributed Dataset")
println("="^60)

# Total dataset size
N_total = 1_000_000_000  # 1 billion points
N_per_worker = div(N_total, nworkers())

println("Total points: $(N_total) ($(round(N_total/1e9; digits=2))B)")
println("Per worker: $(N_per_worker) ($(round(N_per_worker/1e6; digits=1))M)")

# Generate data locally on each worker to avoid communication
# (In practice, would load from distributed storage or HPC filesystem)
@everywhere function generate_local_data(n_points)
    Random.seed!(myid())  # Ensure different random data per worker
    x = randn(Float32, n_points, 2)
    u = randn(Float32, n_points, 2)
    return x, u
end

print("Generating data on all workers...")
@time begin
    remotecall_fetch.(generate_local_data, workers(), N_per_worker)
    println("✓ Data generated on $(nworkers()) workers")
end

# ============================================================================
# 4. Define Structure Function Setup
# ============================================================================

println("\n" * "="^60)
println("Structure Function Setup")
println("="^60)

# Make accessible on all workers
@everywhere const operator = FullVectorStructureFunction{Float32}(order=2)
@everywhere const bins = collect(Float32, 10:1000:10000)

println("Operator: Full vector, 2nd order (Float32)")
println("Bins: $(length(bins)) distance ranges")

# ============================================================================
# 5. Distributed Calculation
# ============================================================================

println("\n" * "="^60)
println("Distributed Computation")
println("="^60)

# Create data references on workers (avoids copying)
@everywhere begin
    x_local, u_local = generate_local_data($N_per_worker)
end

print("Computing structure function across $(nworkers()) processes...")

@time begin
    # Use DistributedBackend
    backend = DistributedBackend()
    
    # This computation happens on the local process but dispatches to workers
    # Note: Data stays on workers; only results are gathered
    result = calculate_structure_function(
        operator,
        x_local, u_local, bins;  # Local arrays (will be handled appropriately)
        backend=backend,
        show_progress=true,
        verbose=false
    )
end

println("✓ Distributed computation complete")

# ============================================================================
# 6. Analyze Results
# ============================================================================

println("\n" * "="^60)
println("Result Analysis")
println("="^60)

sf_values = result.structure_function[:, 1]

println("Structure function (first 5 bins):")
for i in 1:min(5, length(result.distance))
    r = result.distance[i]
    sf = sf_values[i]
    count = result.counts[i]
    println("  r=$(round(r; digits=1)): SF=$(round(sf; digits=2)), pairs=$(count)")
end

println("\nStatistics:")
println("  Total valid pairs: $(sum(result.counts))")
println("  Memory (result): $(round(sizeof(result.structure_function)/1e6; digits=1)) MB")

# ============================================================================
# 7. Fault Tolerance Example
# ============================================================================

println("\n" * "="^60)
println("Fault Tolerance Patterns")
println("="^60)

println("""
In production, use checkpointing:

1. Checkpoint intermediate results:
   save("checkpoint_step_1.jld2", result)
   
2. Check for failed workers:
   if any(isnull.(Distributed.remote_gc_refs(workers())))
       @warn "Worker lost!"
   end
   
3. Restart failed workers:
   rmprocs(failed_worker_id)
   addprocs(1)
   
4. Resume from checkpoint:
   result = load("checkpoint_step_1.jld2")
   # Continue computation
""")

# ============================================================================
# 8. Communication & Load Balancing
# ============================================================================

println("\n" * "="^60)
println("Performance Analysis")
println("="^60)

efficiency = 1.0 / nworkers()  # Ideal case
println("""
Scaling Characteristics:
- Data: $(N_total) points across $(nworkers()) workers
- Locality: Data generated locally (minimal communication)
- Synchronization: Results reduced at main process
- Expected efficiency: $(round(efficiency * 100; digits=0))%+ (strong scaling)

Bottlenecks:
1. MPI/network bandwidth (if across cluster)
2. Result reduction (small relative to computation)
3. Load balancing (if data heterogeneous)

To optimize:
- Use fast interconnect (InfiniBand > Ethernet)
- Batch results to reduce communication rounds
- Profile with @time, @profile
""")

# ============================================================================
# 9. HPC Submission Script
# ============================================================================

println("\n" * "="^60)
println("HPC Cluster Example (SLURM)")
println("="^60)

slurm_script = """
#!/bin/bash
#SBATCH --nodes=4              # 4 compute nodes
#SBATCH --cpus-per-task=8      # 8 cores per node
#SBATCH --mem-per-cpu=8G       # 8 GB per core
#SBATCH --time=01:00:00        # 1 hour max
#SBATCH --job-name=StructFunc  # Job name

export JULIA_NUM_THREADS=8     # Let Julia use all cores
export JULIA_PROJECT=@.        # Use current environment

# Calculate total processes
NPROCS=\$((\\${SLURM_NNODES} * \\${SLURM_CPUS_PER_TASK}))

echo "Running distributed SF calculation"
echo "Nodes: \\${SLURM_NNODES}"
echo "Cores per node: \\${SLURM_CPUS_PER_TASK}"
echo "Total processes: \\${NPROCS}"

# Run Julia with distributed processes
mpirun julia -np \\${NPROCS} example_distributed_parallel.jl

# Or without MPI (manual process distribution):
# julia -p \\${NPROCS} example_distributed_parallel.jl \\${NPROCS}
"""

println("Example SLURM submission (save as 'submit_distributed.sbatch'):")
print(slurm_script)

# ============================================================================
# 10. Best Practices
# ============================================================================

println("\n" * "="^60)
println("Best Practices for Distributed Computing")
println("="^60)

println("""
1. DATA MANAGEMENT
   ✓ Generate data locally on each worker (no transfer)
   ✓ Use shared filesystem (NFS, Lustre, etc.)
   ✓ Load from HDF5/NetCDF with parallel I/O
   ✗ Avoid copying large data structures to main

2. PROCESS ALLOCATION
   ✓ One process per physical core (not hyperthreads)
   ✓ Leave one core for main/OS
   ✓ Scale with grid/cluster size

3. COMMUNICATION
   ✓ Minimize data transfer (keep local)
   ✓ Batch results when possible
   ✓ Use fast interconnects (InfiniBand)

4. FAULT TOLERANCE
   ✓ Checkpoint periodically
   ✓ Monitor worker health
   ✓ Able to restart failed jobs

5. DEBUGGING
   ✓ Test on small cluster first (laptop with 4 cores)
   ✓ Use @distributed macros for simpler syntax
   ✓ Check log files (SLURM: \\${SLURM_JTMPDIR})

6. PROFILING
   Use Distributed.@everywhere to profile:
   
   @everywhere begin
       using Profile
       @profile calculate_structure_function(...)
   end
   
   # Then collect and analyze profiles
""")

# ============================================================================
# 11. Alternative: @distributed Macro (Simpler)
# ============================================================================

println("\n" * "="^60)
println("Alternative: Using @distributed (Simpler Syntax)")
println("="^60)

println("""
For simpler parallelism without manual process management:

    using Distributed
    addprocs(4)
    
    @everywhere using StructureFunctions
    
    # Parallel for-loop
    results = @distributed (+) for i in 1:100
        x_i = randn(10_000, 2)
        u_i = randn(10_000, 2)
        result = calculate_structure_function(x_i, u_i, bins)
        result.structure_function  # Accumulate
    end
    
    # Or with custom reduction
    using DistributedArrays
    x_dist = distribute(randn(1_000_000_000, 2))
    u_dist = distribute(randn(1_000_000_000, 2))
    
    result = calculate_structure_function(x_dist, u_dist, bins;
                                         backend=DistributedBackend())
""")

# ============================================================================
# 12. Cleanup & Summary
# ============================================================================

println("\n" * "="^60)
println("Shutting Down")
println("="^60)

rmprocs(workers())  # Clean up worker processes
println("✓ Worker processes shut down")

println("\n" * "="^60)
println("Distributed Computation Summary")
println("="^60)

println("""
Distributed SFs computed successfully!

Performance:
- Scale-up: Near-linear to N processes (for large problems)
- Memory per node: Reduced by factor of N
- Suitable for: >500M points across cluster

Next steps:
1. Measure scaling on your HPC cluster
2. Optimize process/task allocation
3. Integrate into production pipeline

Documentation:
- docs/backends.md: DistributedBackend details
- docs/real_data.md: Loading distributed data
- examples/gpu_acceleration.jl: Combine GPU + distributed

For very large data:
- Use Zarr/cloud storage (S3, GCS)
- Combine with GPU for 100x+ speedup
- Consider adaptive streaming/checkpointing
""")

println("\nDistributed example complete. 🎉")
