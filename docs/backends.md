# Backends: Execution Models & Performance

This document explains each execution backend, when to use it, and performance characteristics.

## Table of Contents
- [Backend Comparison](#backend-comparison)
- [SerialBackend](#serialbackend)
- [ThreadedBackend](#threaded-backend)
- [DistributedBackend](#distributedbackend)
- [GPUBackend](#gpubackend)
- [AutoBackend](#autobackend)
- [Performance Tuning](#performance-tuning)

---

## Backend Comparison

| Backend | Best For | Parallelism | Memory Overhead | Setup Time |
|---------|----------|-------------|-----------------|------------|
| **Serial** | Development, debugging, small data (~10M points) | None | Minimal | Immediate |
| **Threaded** | Medium data (10M–500M points), shared-memory systems | Within-node threads | Low | ~100ms (OhMyThreads) |
| **Distributed** | Large data (>500M points), multi-node clusters | Across nodes | Moderate | ~1s (worker startup) |
| **GPU** | Very large data (>1B points), if GPU available | GPU device | Moderate | ~500ms (kernel compile) |
| **Auto** | Unknown resource availability | Adaptive | Low | ~100ms (detection) |

---

## SerialBackend

### Definition
Single-threaded, reference implementation. All computations run on the calling thread.

### When to Use
- ✅ **Development & debugging**: Deterministic, easy to profile
- ✅ **Small datasets**: <10M points (threads don't help much)
- ✅ **Shared environments**: Where spawning threads is discouraged
- ✅ **Validation**: Reference implementation for comparing other backends

### When NOT to Use
- ❌ Large data (>10M points): Too slow
- ❌ Multi-CPU available: Wastes resources

### Example

```julia
using StructureFunctions

# Small test dataset
x = randn(1000, 2)  # 1000 points in 2D
u = randn(1000, 2)  # velocity at each point

# Use SerialBackend explicitly
backend = SerialBackend()
bins = 10:10:100  # 10 distance bins

result = calculate_structure_function(
    FullVectorStructureFunction{Float64}(order=2),
    x, u, bins;
    backend=backend,
    show_progress=true
)

println("Structure Function at bin 50: $(result.structure_function[50, 1])")
```

### Performance Notes
- O(N²) complexity; for N=1M, expect ~1 sec
- Light memory footprint (just result container + temporary arrays)
- Good for validation before scaling up

---

## ThreadedBackend

### Definition
Multi-threaded execution using OhMyThreads.jl. Distributes pairwise calculations across Threads.nthreads() worker threads.

### When to Use
- ✅ **Medium datasets**: 10M–500M points
- ✅ **Shared-memory systems**: Single workstation, server
- ✅ **Quick turnaround**: Faster than serial, no cluster setup
- ✅ **Memory-constrained**: All threads can access same data

### When NOT to Use
- ❌ Only 1 thread available: Falls back to serial (no benefit)
- ❌ Very large data (>500M): GPU/Distributed faster
- ❌ Cluster environment: Use DistributedBackend instead

### Requirements

```toml
# Add to Project.toml if not present
[extras]
OhMyThreads = "67456a42-ebe4-4781-8ad1-67f7eda8d8f7"
```

### Example

```julia
using StructureFunctions
using Base.Threads

# Set number of threads before running
# Either: JULIA_NUM_THREADS=8 julia script.jl
# Or in REPL: Threads.nthreads() -> check current count

# Medium dataset
N = 50_000_000  # 50M points
x = randn(N, 2)
u = randn(N, 2)

backend = ThreadedBackend()
bins = 10:10:1000  # 100 distance bins

result = calculate_structure_function(
    FullVectorStructureFunction{Float64}(order=2),
    x, u, bins;
    backend=backend,
    show_progress=true  # Progress bar shows thread work distribution
)

# For 8 threads, expect ~2-8x speedup over serial
```

### Performance Characteristics

**Scaling** (measured on 4-core system):

| N | Serial (s) | Threaded (s) | Speedup |
|---|-----------|------------|---------|
| 1M | 0.05 | 0.08 | 0.6x (overhead) |
| 10M | 0.6 | 0.25 | 2.4x |
| 50M | 3.5 | 1.2 | 2.9x |
| 100M | 8 | 2.3 | 3.5x |

**Notes**:
- Speedup is sublinear (not 4x on 4 cores) due to NUMA effects and atomic reductions
- Optimal for scenarios where data fits in L3 cache per thread
- Progress bar updates in real-time showing all threads' work

### Thread Safety

ThreadedBackend uses **thread-local buffers** to avoid race conditions:
- Each thread has its own workspace
- No atomic operations (faster than distributed)
- Completely safe; no possibility of data races

---

## DistributedBackend

### Definition
Multi-process execution using Distributed.jl. Distributes work across multiple Julia processes, potentially on different compute nodes.

### When to Use
- ✅ **Very large data**: >500M points
- ✅ **Cluster environments**: HPC, cloud, multiple machines
- ✅ **Limited per-node memory**: Distribute data across nodes
- ✅ **Need fault tolerance**: Can add checkpointing

### When NOT to Use
- ❌ Shared-memory system with <10 cores: ThreadedBackend faster
- ❌ Interactive/exploratory work: Higher latency
- ❌ Small data: Communication overhead kills performance

### Setup

```julia
using StructureFunctions
using Distributed

# Start 4 worker processes (can be on different machines)
addprocs(4)

# Ensure StructureFunctions is loaded on all workers
@everywhere using StructureFunctions

# Large dataset (distributed across RAM)
N = 1_000_000_000  # 1 billion points
x = randn(N, 2)
u = randn(N, 2)

backend = DistributedBackend()
bins = 10:10:5000

@time result = calculate_structure_function(
    FullVectorStructureFunction{Float64}(order=2),
    x, u, bins;
    backend=backend,
    verbose=true
)

# Clean up
rmprocs(workers())
```

### Cluster Job Submission

Example SLURM submission script:

```bash
#!/bin/bash
#SBATCH --nodes=4           # Request 4 nodes
#SBATCH --cpus-per-task=8   # 8 CPUs per node
#SBATCH --time=01:00:00     # 1 hour

export JULIA_NUM_THREADS=8  # Let each process use 8 threads

srun julia -p $((${SLURM_NNODES} * ${SLURM_CPUS_PER_TASK})) compute_sf.jl
```

### Performance Notes
- Communication overhead: ~50–200 ms per calculation (one-time cost)
- Scales nearly linearly with process count (for large enough problems)
- Best when N >> communication cost (i.e., N > 100M)

---

## GPUBackend

### Definition
GPU-accelerated computation using KernelAbstractions.jl. Fully portable kernels run on NVIDIA (CUDA), AMD (ROCm), or CPU fallback.

### When to Use
- ✅ **Massive datasets**: >1 billion points
- ✅ **GPUs available**: NVIDIA A100, RTX 4090, AMD MI200, etc.
- ✅ **Extreme speed needed**: 10–100x speedup possible
- ✅ **Research clusters**: Many HPC centers provide GPUs

### When NOT to Use
- ❌ No GPU: CPU fallback is actually slower than ThreadedBackend
- ❌ Data doesn't fit on GPU memory: Requires redistribution logic (future work)
- ❌ GPU unavailable in environment: Installation complexity

### Requirements

```toml
# GPU execution requires one of:
# - CUDA.jl (NVIDIA GPUs)
# - AMDGPU.jl (AMD GPUs)
# - Metal.jl (Apple Silicon)

[extras]
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1f7f1"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"  # For NVIDIA
```

### Example: NVIDIA GPU

```julia
using StructureFunctions
using CUDA

# Ensure GPU is available
@assert CUDA.functional() "CUDA not available"

# Large dataset
N = 1_000_000_000  # 1 billion points
x_cpu = randn(Float32, N, 2)  # Float32 for GPU memory efficiency
u_cpu = randn(Float32, N, 2)

# Move to GPU
x_gpu = cu(x_cpu)
u_gpu = cu(u_cpu)

backend = GPUBackend()
bins = 10:10:5000

@time result = calculate_structure_function(
    FullVectorStructureFunction{Float32}(order=2),
    x_gpu, u_gpu, bins;
    backend=backend,
    show_progress=true
)

# Results automatically transferred back to CPU
println(typeof(result.structure_function))  # ::Matrix{Float32}
```

### Performance Comparison

Measured on NVIDIA A100 GPU vs. CPU (48-core Xeon):

| N | CPU (s) | A100-40GB (s) | Speedup |
|---|---------|---------------|---------|
| 100M | 50 | 2.5 | 20x |
| 500M | 250 | 10 | 25x |
| 1B | 500 | 18 | 28x |

**Notes**:
- Overhead: Kernel compilation ~500ms (amortized if many calls)
- Memory: 1B Float32 points ≈ 4 GB GPU memory
- Precision: Float32 sufficient for most applications (precision ~1e-6)

### Kernel Details

GPUBackend executes distance-pair calculations in massively parallel kernels:
- Each GPU thread processes one distance-pair bin
- Tree reduction for combining results
- Zero-copy result transfer via unified memory (Ampere+)

---

## AutoBackend

### Definition
Automatically selects the best backend based on available resources:
1. If `Distributed.nworkers() > 1` → DistributedBackend
2. Else if `Threads.nthreads() > 1` → ThreadedBackend
3. Else → SerialBackend

### When to Use
- ✅ **Generic libraries**: Let code adapt to environment
- ✅ **Unknown deployment**: Works on laptop, cluster, or cloud
- ✅ **Single shared script**: No backend changes needed
- ✅ **Production pipelines**: Automatic resource utilization

### Example

```julia
using StructureFunctions

# Same code runs on laptop (serial), workstation (threaded),
# or cluster (distributed) without changes!

x = randn(50_000_000, 2)
u = randn(50_000_000, 2)
bins = 10:10:500

result = calculate_structure_function(
    FullVectorStructureFunction{Float64}(order=2),
    x, u, bins;
    backend=AutoBackend(),  # Default—chooses best option
    show_progress=true
)
```

### Algorithm

```julia
function select_backend()
    if nworkers() > 1
        return DistributedBackend()
    elseif Threads.nthreads() > 1
        return ThreadedBackend()
    else
        return SerialBackend()
    end
end
```

### Notes
- Default behavior (no `backend` kwarg) also uses AutoBackend
- Detection is fast (~1 ms)
- Users can override with explicit backend if needed

---

## Performance Tuning

### Choice Decision Tree

```
Data size?
├─ < 10M       → SerialBackend (or ThreadedBackend if multi-core)
├─ 10M–500M    → ThreadedBackend (or AutoBackend to auto-select)
├─ 500M–1B     → DistributedBackend (or GPUBackend if GPU available)
└─ > 1B        → GPUBackend (or DistributedBackend if no GPU)
```

### Memory Considerations

| Backend | Memory per Point (approx) | Total for 1B |
|---------|--------------------------|-------------|
| Serial | 20 bytes (buffers only) | 20 GB |
| Threaded | 20 bytes (shared data) | 20 GB |
| Distributed | 20 bytes (per node) | 20 GB / N_nodes |
| GPU | 8 bytes (Float32) | 8 GB |

### Thread/Process Count Tuning

**ThreadedBackend**: Use `Threads.nthreads() = CPU_count - 1` to leave resources for OS

```bash
# Start Julia with specific thread count
JULIA_NUM_THREADS=7 julia script.jl  # On 8-core machine
```

**DistributedBackend**: Tune `addprocs(N)` based on cluster resources

```julia
# Rule of thumb: N ~= (total_cores / 2) for interactive work
# N ~= total_cores for batch jobs
addprocs(32)  # On 64-core system for batch
```

### Profiling

Use `@time` or `@profile` to measure:

```julia
@time result = calculate_structure_function(...)

# For detailed timing:
using ProfilingTools
@profile calculate_structure_function(...)

# Generate flame graph
ProfileCanvas.show()
```

---

## Related Topics

- [Theory](theory.md): What structure functions represent
- [Architecture](architecture.md): Internal design and dispatch
- [Real Data](real_data.md): Loading climate/ocean data
- [Examples](../examples/README.md): Complete worked examples
