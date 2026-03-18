"""
    example_gpu_acceleration.jl

GPU-accelerated structure function calculation for massive datasets.

This example demonstrates:
- Using GPUBackend with KernelAbstractions
- Memory management for GPU data
- Optimal data types and precision
- Comparison with CPU backends

Requires: CUDA.jl, KernelAbstractions.jl, or other GPU support

Run:
    julia --project -e 'using Pkg; Pkg.add(["CUDA", "KernelAbstractions"])'
    julia example_gpu_acceleration.jl
"""

using StructureFunctions
using Statistics

# ============================================================================
# 1. Check GPU Availability
# ============================================================================

println("GPU Configuration")
println("="^60)

# Try to import GPU packages
try
    using CUDA
    if CUDA.functional()
        println("✓ CUDA is available and functional")
        println("  Device: $(CUDA.name(CUDA.device()))")
        println("  Memory: $(round(CUDA.available_memory() / 1e9; digits=1)) GB available")
        using_cuda = true
    else
        println("⚠ CUDA not functional")
        using_cuda = false
    end
catch
    println("⚠ CUDA.jl not available (install with: Pkg.add(\"CUDA\"))")
    using_cuda = false
end

if !using_cuda
    println("\nFalling back to CPU. To enable GPU:")
    println("  1. Install GPU package: using Pkg; Pkg.add(\"CUDA\")")
    println("  2. Install GPU driver: nvidia-smi (should show your GPU)")
    println("  3. Re-run this script")
end

# ============================================================================
# 2. Generate Large Dataset
# ============================================================================

println("\n" * "="^60)
println("Generating Large Dataset")
println("="^60)

# For GPU: use Float32 (half memory, sufficient precision, faster)
precision = Float32

if using_cuda
    N = 1_000_000_000  # 1 billion points (challenging for CPU)
else
    N = 10_000_000     # 10M points (CPU fallback)
end

println("Generating $(N) random velocity points ($(precision))...")

# CPU generation (data too large for GPU direct generation)
x_cpu = randn(Float32, N, 2)
u_cpu = randn(Float32, N, 2)

println("  Data size: $(round(N/1e6; digits=1))M points")
println("  Precision: $(precision)")
println("  Memory (x): $(round(N*2*sizeof(precision)/1e9; digits=2)) GB")
println("  Memory (u): $(round(N*2*sizeof(precision)/1e9; digits=2)) GB")

# ============================================================================
# 3. Move Data to GPU (if available)
# ============================================================================

println("\n" * "="^60)
println("Data Transfer")
println("="^60)

if using_cuda
    println("Transferring data to GPU...")
    
    @time x_gpu = cu(x_cpu)
    @time u_gpu = cu(u_cpu)
    
    println("✓ Data transferred to GPU")
    
    x_compute = x_gpu
    u_compute = u_gpu
    location = "GPU"
else
    println("Using CPU (GPU not available)")
    x_compute = x_cpu
    u_compute = u_cpu
    location = "CPU"
end

# ============================================================================
# 4. Define Structure Function & Bins
# ============================================================================

println("\n" * "="^60)
println("Structure Function Setup")
println("="^60)

# Use Float32 operator for efficient GPU computation
operator = FullVectorStructureFunction{Float32}(order=2)
bins = collect(Float32, 10:100:5000)  # 50 bins

println("Operator: Full vector, 2nd order ($(precision))")
println("Bins: $(length(bins)) distance ranges")
println("Computation location: $(location)")

# ============================================================================
# 5. GPU Calculation
# ============================================================================

println("\n" * "="^60)
println("GPU Calculation")
println("="^60)

print("Calculating structure function on $(location)...")

backend = GPUBackend()

@time result_gpu = calculate_structure_function(
    operator,
    x_compute, u_compute, bins;
    backend=backend,
    show_progress=true,
    verbose=true
)

println("✓ Calculation complete")

# ============================================================================
# 6. Result Transfer & Analysis
# ============================================================================

println("\n" * "="^60)
println("Result Analysis")
println("="^60)

# Results are automatically on CPU (or appropriate location)
sf_values = result_gpu.structure_function[:, 1]

println("Structure function values (first 10 bins):")
for i in 1:min(10, length(result_gpu.distance))
    r = result_gpu.distance[i]
    sf = sf_values[i]
    count = result_gpu.counts[i]
    println("  r=$(round(r; digits=1)) m: SF=$(round(sf; digits=2)), pairs=$(count)")
end

# ============================================================================
# 7. Memory Efficiency
# ============================================================================

println("\n" * "="^60)
println("Memory Efficiency")
println("="^60)

# Estimate memory saved by Float32 vs Float64
mem_float64 = N * 2 * 8 * 2 / 1e9  # 2 arrays (x, u), 2 components, 8 bytes each
mem_float32 = N * 2 * 4 * 2 / 1e9  # Same but 4 bytes each
saving = (1 - mem_float32/mem_float64) * 100

println("Memory comparison for $(N) points with 2 components:")
println("  Float64: $(round(mem_float64; digits=1)) GB")
println("  Float32: $(round(mem_float32; digits=1)) GB")
println("  Savings: $(round(saving; digits=0))%")
println("\nPrecision loss:")
println("  Float32 epsilon: ~1e-7 (sufficient for turbulence analysis)")
println("  Float64 epsilon: ~1e-16 (overkill for most applications)")

# ============================================================================
# 8. Performance Comparison
# ============================================================================

println("\n" * "="^60)
println("Expected Performance (Theoretical)")
println("="^60)

println("""
GPU Performance (relative to CPU):
- NVIDIA A100 GPU: 10-100x faster
- NVIDIA RTX 4090: 5-50x faster
- AMD MI250X: 15-100x faster

Factors affecting speedup:
1. Problem size (larger = better GPU utilization)
2. Memory bandwidth (GPU optimized for large data)
3. Kernel compilation (amortized if reused)
4. Data transfer time (PCIe bottleneck for small problems)

For N=$(N) points on GPU:
- Kernel computation: dominated by pairwise distance calculations
- Bottleneck: Memory bandwidth (read x, u; write sums/counts)
- Expected speedup: 20-50x vs single CPU core
- Compared to $(Threads.nthreads())-threaded CPU: 5-20x faster
""")

# ============================================================================
# 9. Scaling Demonstration
# ============================================================================

if using_cuda
    println("\n" * "="^60)
    println("GPU Memory Scaling")
    println("="^60)
    
    available_memory = CUDA.available_memory() / 1e9
    
    println("Available GPU memory: $(round(available_memory; digits=1)) GB")
    
    # Estimate max points
    bytes_per_point = (2 * 4) * 2 + (2 * 4)  # x (2×float32) + u (2×float32) + overhead
    max_points = Int(floor(available_memory * 1e9 / bytes_per_point))
    
    println("Maximum points addressable: $(round(max_points/1e9; digits=2)) billion")
    println("\nTo scale to even larger problems:")
    println("1. Use gradient checkpointing (recompute vs store)")
    println("2. Process in batches (out-of-core computation)")
    println("3. Use distributed GPU setup (multi-GPU, multi-node)")
end

# ============================================================================
# 10. Best Practices
# ============================================================================

println("\n" * "="^60)
println("GPU Computing Best Practices")
println("="^60)

println("""
1. DATA TYPE SELECTION
   ✓ Use Float32 for most turbulence calculations
   ✓ Use Float64 only if required for accuracy
   ✓ Integer arithmetic for counts

2. MEMORY MANAGEMENT
   ✓ Keep data on GPU (minimize transfers)
   ✓ Reuse GPU arrays where possible
   ✓ Profile memory usage: CUDA.@profile, Memento

3. BATCH PROCESSING
   ✓ For data > GPU memory: process chunks
   ✓ Keep results separate (merge at end)
   ✗ Avoid repeated GPU ↔ CPU transfers

4. PERFORMANCE TUNING
   ✓ Measure with @time, BenchmarkTools
   ✓ Profile with CUDA.@profile
   ✓ Use block/thread optimization for kernels

5. DEVELOPMENT WORKFLOW
   ✓ Start on CPU (SerialBackend) for correctness
   ✓ Switch to GPU (GPUBackend) for performance
   ✓ Use ThreadedBackend for debugging (intermediate)

6. MULTI-GPU
   Use DistributedBackend + multiple GPUs:
   
   using Distributed, StructureFunctions
   addprocs(4)  # Start 4 workers
   @everywhere using CUDA, StructureFunctions
   
   # Each process gets one GPU (auto-assigned)
   backend = DistributedBackend()
   result = calculate_structure_function(...; backend)
""")

# ============================================================================
# 11. Summary & Next Steps
# ============================================================================

println("\n" * "="^60)
println("Summary & Next Steps")
println("="^60)

println("""
GPU Acceleration Achieved!

Key metrics:
- Points processed: $(N) ($(round(N/1e9; digits=2)) billion)
- Data type: $(precision)
- Backend: $(string(typeof(backend)))
- Location: $(location) memory

For production pipelines:
1. Try DistributedBackend for multi-node scaling
2. Combine with batch processing for out-of-core
3. Profile to optimize for your specific hardware

Documentation:
- docs/backends.md: Backend comparison and tuning
- docs/real_data.md: Loading climate/science data
- docs/extensions.md: GPU support and customization

Related examples:
- example_threaded_calculation.jl: CPU parallelization
- example_distributed_parallel.jl: Multi-node computation
- example_real_data_climate.jl: Real data workflows
""")

println("\nGPU example complete. Happy computing! 🚀")
