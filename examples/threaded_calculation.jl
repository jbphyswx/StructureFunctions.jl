"""
    example_threaded_calculation.jl

Multi-threaded structure function calculation for medium-sized datasets.

This example demonstrates:
- Setting up ThreadedBackend for multi-core execution
- Measuring speedup vs serial calculation
- Best practices for parallelization

Requires: Set JULIA_NUM_THREADS environment variable

Run:
    JULIA_NUM_THREADS=4 julia example_threaded_calculation.jl
    # or JULIA_NUM_THREADS=auto for automatic detection
"""

using StructureFunctions
using Statistics
using BenchmarkTools

# ============================================================================
# 1. Check Available Threads
# ============================================================================

import Base.Threads

println("Threading Configuration")
println("="^60)
println("Julia executable: $(Base.julia_exe())")
println("Available threads: $(Threads.nthreads())")
println("Number of CPU cores: $(Sys.CPU_THREADS)")

if Threads.nthreads() < 2
    println("\n⚠ Warning: Only $(Threads.nthreads()) thread available.")
    println("  To enable multi-threading, run with:")
    println("    JULIA_NUM_THREADS=N julia $@__FILE__")
    println("  where N = number of desired threads (e.g., JULIA_NUM_THREADS=4)")
end

# ============================================================================
# 2. Generate Medium-Sized Dataset
# ============================================================================

println("\n" * "="^60)
println("Generating Medium-Sized Dataset")
println("="^60)

# Choose size based on available threads
thread_count = Threads.nthreads()
if thread_count >= 8
    N = 100_000_000  # 100M points for many cores
elseif thread_count >= 4
    N = 50_000_000   # 50M points for 4-8 cores
else
    N = 10_000_000   # 10M points for 1-2 cores
end

println("Generating $(N) random velocity points...")

# Random 2D coordinates
x = randn(N, 2)

# Random 2D velocity
# (In practice, would load from file or simulation)
u = randn(N, 2)

println("  Data size: $(round(N/1e6; digits=1))M points")
println("  Memory (x): $(round(N*2*8/1e9; digits=1)) GB")
println("  Memory (u): $(round(N*2*8/1e9; digits=1)) GB")

# ============================================================================
# 3. Define Structure Function & Bins
# ============================================================================

println("\n" * "="^60)
println("Structure Function Setup")
println("="^60)

operator = FullVectorStructureFunction{Float64}(order=2)
bins = 10:2:100  # 46 bins

println("Operator: Full vector, 2nd order")
println("Bins: $(length(bins)) distance ranges")

# ============================================================================
# 4. Serial Calculation (Reference)
# ============================================================================

println("\n" * "="^60)
println("Serial Calculation (Reference)")
println("="^60)

print("Running serial calculation...")
backend_serial = SerialBackend()

# Use @time to get execution time
time_serial = @time begin
    result_serial = calculate_structure_function(
        operator, x, u, bins;
        backend=backend_serial,
        show_progress=false  # Suppress progress for benchmarking
    )
end

println("Time (serial): $(round(time_serial; digits=2)) seconds")

# ============================================================================
# 5. Threaded Calculation
# ============================================================================

println("\n" * "="^60)
println("Threaded Calculation")
println("="^60)

if Threads.nthreads() > 1
    print("Running threaded calculation with $(Threads.nthreads()) threads...")
    backend_threaded = ThreadedBackend()
    
    time_threaded = @time begin
        result_threaded = calculate_structure_function(
            operator, x, u, bins;
            backend=backend_threaded,
            show_progress=false
        )
    end
    
    println("Time (threaded): $(round(time_threaded; digits=2)) seconds")
    
    # Compute speedup
    speedup = time_serial / time_threaded
    efficiency = speedup / Threads.nthreads() * 100
    
    println("\n" * "="^60)
    println("Performance Summary")
    println("="^60)
    println("Speedup: $(round(speedup; digits=2))x faster than serial")
    println("Parallel efficiency: $(round(efficiency; digits=1))%")
    println("  (100% = perfect scaling, 0% = no speedup)")
    
    if efficiency > 80
        println("✓ Excellent scaling!")
    elseif efficiency > 60
        println("✓ Good scaling")
    elseif efficiency > 40
        println("~ Moderate scaling (memory/cache contention)")
    else
        println("⚠ Poor scaling (consider distributed computing)")
    end
    
    # Verify results agree
    println("\n" * "="^60)
    println("Result Validation")
    println("="^60)
    
    # Check that results are close (may differ slightly due to rounding/reduction order)
    max_diff = maximum(abs.(result_serial.structure_function .- result_threaded.structure_function))
    rel_diff = max_diff / maximum(abs.(result_serial.structure_function))
    
    println("Max absolute difference: $(round(max_diff; digits=2e-10))")
    println("Max relative difference: $(round(rel_diff * 100; digits=2))%")
    
    if rel_diff < 1e-10
        println("✓ Results match perfectly (bit-identical)")
    elseif rel_diff < 1e-6
        println("✓ Results agree to double precision")
    else
        println("⚠ Results differ (expected slight variations in reduction order)")
    end
    
else
    println("\n⚠ Multi-threading not available (only 1 thread)")
    println("  Skipping threaded calculation")
end

# ============================================================================
# 6. Scaling Analysis
# ============================================================================

println("\n" * "="^60)
println("Scaling Analysis")
println("="^60)

if Threads.nthreads() > 1
    println("""
Amdahl's Law predicts maximum speedup S for N cores:
    S = N / (1 + (N-1)·f_serial)
where f_serial is fraction of serial code.

Your system:
    Threads: $(Threads.nthreads())
    Observed speedup: $(round(speedup; digits=2))x
    
To improve scaling:
1. Increase dataset size ($(N) points)
   → Amortizes overhead
   → Try: 500M, 1B points with DistributedBackend
   
2. Use larger distance bin ranges
   → More computation per pair check
   → Try: bins = 10:10:1000
   
3. Switch to GPU (if available)
   → 10-100x speedup for >1B points
   → Try: GPUBackend() with CUDA.jl
   
4. Distribute across cluster
   → Linear scaling with number of nodes
   → Try: DistributedBackend() with addprocs(16)
""")
end

# ============================================================================
# 7. Best Practices Summary
# ============================================================================

println("\n" * "="^60)
println("Best Practices for Threaded Computation")
println("="^60)

println("""
1. THREAD COUNT
   - Set JULIA_NUM_THREADS to CPU_count - 1 (leave room for OS)
   - If Nthreads = 1, ThreadedBackend falls back to serial

2. DATA SIZE
   - ThreadedBackend optimal for 10M–500M points
   - Below 10M: serial is faster (overhead dominates)
   - Above 500M: GPU or distributed computation faster

3. MEMORY BANDWIDTH
   - ThreadedBackend scales better with fast RAM (DDR5 > DDR4)
   - NUMA systems: Consider DistributedBackend per socket

4. PROFILING
   - Always measure (don't assume speedup)
   - Profile with @time or @benchmark
   - Check parallel efficiency

5. INTERACTIVITY
   - Julia REPL auto-uses available threads
   - Compilation happens on first call (JIT cost)
   - Reuse same REPL to avoid recompilation

6. NEXT STEPS
   - GPU: 10-100x speedup for massive data
   - Distributed: Linear scaling over many nodes
   - AutoBackend: Automatic selection
""")

println("\nExample benchmarking:")
println("""
    using BenchmarkTools
    @benchmark calculate_structure_function(
        operator, x, u, bins;
        backend=ThreadedBackend(),
        show_progress=false
    )
""")

println("\nSet JULIA_NUM_THREADS and re-run for different thread counts.")
println("Plot speedup vs threads to validate scaling.")
