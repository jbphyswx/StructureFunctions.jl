"""
    examples/README.md

Worked examples and tutorials for StructureFunctions.jl
"""

# Examples: Getting Started with StructureFunctions.jl

This directory contains runnable examples demonstrating different aspects of structure function computation in StructureFunctions.jl.

## Quick Start

### 1. Simple 2D Calculation (`simple_2d.jl`)

**Best for**: First-time users, understanding basics

```bash
julia examples/simple_2d.jl
```

**What it does**:
- Generates synthetic 2D turbulent velocity data (256×256 grid)
- Computes 2nd-order structure functions
- Validates K41 scaling ($S_2(r) \sim r^{2/3}$)
- Produces visualization (plot saved)

**Key concepts**:
- Regular grids and structure functions
- Spectral scaling laws
- Visualization of results

**Learn next**: `docs/theory.md` for physics background

---

### 2. Threaded Parallelization (`threaded_calculation.jl`)

**Best for**: Medium datasets (10M–500M points), multi-core machines

```bash
# Set thread count
JULIA_NUM_THREADS=4 julia examples/threaded_calculation.jl

# Or let Julia auto-detect
JULIA_NUM_THREADS=auto julia examples/threaded_calculation.jl
```

**What it does**:
- Generates 50M points (scalable based on CPU cores)
- Compares serial vs threaded execution
- Measures speedup and parallel efficiency
- Demonstrates best practices

**Key metrics**:
- Speedup vs serial: typically 2–8x on 4–32 cores
- Efficiency: depends on memory bandwidth
- Optimal for cache-resident data

**Advanced**: Try different dataset sizes; plot speedup curve vs thread count

**Learn next**: `docs/backends.md` for backend comparison

---

### 3. GPU Acceleration (`gpu_acceleration.jl`)

**Best for**: GPU hardware or KA.CPU smoke test; workspace reuse

```bash
julia --project=examples examples/gpu_acceleration.jl
# With CUDA:
julia --project=gpu examples/gpu_acceleration.jl
```

**What it does**:
- Computes longitudinal 2nd-order SF on ~2k 3D points
- Compares fresh alloc vs `GPUSFWorkspace`
- Falls back to `KA.CPU()` when CUDA unavailable

**Learn next**: [`docs/gpu.md`](../docs/gpu.md), [`docs/backends.md#gpubackend`](../docs/backends.md#gpubackend)

---

### 3b. GPU Time Slices (`gpu_time_slices.jl`)

**Best for**: Time-series / batch slice API

```bash
julia --project=examples examples/gpu_time_slices.jl
```

**What it does**:
- Builds `(3, N, T)` batch and calls `gpu_calculate_structure_function_slices!`
- Validates slice 1 against serial CPU

---

### 4. Distributed Computing (`distributed_parallel.jl`)

**Best for**: Massive datasets across clusters (>1B points, multi-node)

```bash
# Local cluster (spawn 4 processes)
julia examples/distributed_parallel.jl

# Custom process count
julia examples/distributed_parallel.jl 8

# Or via Distributed.jl
julia -p 8 examples/distributed_parallel.jl
```

**What it does**:
- Starts multiple Julia processes
- Distributes data locally (no copy)
- Computes structure functions in parallel
- Demonstrates SLURM submission script

**HPC Usage**:
```bash
# Save the SLURM template:
# sbatch submit_distributed.sbatch

# Scales to 100s of processes on HPC clusters
```

**Scaling characteristics**:
- Near-linear speedup with N nodes (for large problems)
- Suitable for >500M points
- Minimal communication overhead if data is local

**Learn next**: `docs/backends.md#distributedbackend`

---

## Advanced Workflows

### Combining Backends

**Scenario**: Process independent array slices with GPU, aggregate across time

```julia
using StructureFunctions

for t in 1:n_time_steps
    x_t = @view x[:, :, t]
    u_t = @view u[:, :, t]
    result_t = calculate_structure_function(sf, x_t, u_t, bins; backend = GPUBackend())
end
```

### Multi-Scale Nested Analysis

**Scenario**: Analyze turbulence at multiple length scales

```julia
# Coarse grid: long-range structure
bins_coarse = 100:500:100_000
result_coarse = calculate_structure_function(x, u, bins_coarse)

# Fine grid: local correlation
bins_fine = 1:0.5:100
result_fine = calculate_structure_function(x, u, bins_fine)

# Compare scaling exponents
alpha_coarse = estimate_exponent(result_coarse)
alpha_fine = estimate_exponent(result_fine)
```

---

## Running All Examples

```bash
# Prerequisites
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run all in sequence
for script in simple_2d threaded_calculation gpu_acceleration distributed_parallel; do
    echo "Running example: $script"
    julia examples/${script}.jl
done
```

---

## Performance Expectations

| Example | Input Size | Time | Backend |
|---------|-----------|------|---------|
| simple_2d | 65K pts | ~1 sec | Serial |
| threaded_calculation | 50M pts | ~1 sec | Threaded (4 cores) |
| gpu_acceleration | 1B pts | ~20 sec | GPU (A100) |
| distributed_parallel | 1B pts | ~30 sec | Multi-process |

*Actual times vary by hardware. Use `@time` to measure.*

---

## Choosing an Example to Start With

### I want to...

- **Learn the basics** → `simple_2d.jl`
- **Speed up my computation** → `threaded_calculation.jl`
- **Use my GPU** → `gpu_acceleration.jl`
- **Scale to a supercomputer** → `distributed_parallel.jl`
- **Combine multiple approaches** → Mix the examples!

---

## Common Issues & Solutions

### "ThreadedBackend not found"
```julia
julia> ThreadedBackend()
ERROR: UndefVarError: ThreadedBackend not defined

# Solution: Load OhMyThreads
using OhMyThreads  # Triggers extension
```

### "GPU out of memory"
```julia
# Solution: Reduce dataset size or use Float32
x = Float32.(x)  # Half the memory
u = Float32.(u)

result = calculate_structure_function(x, u, bins; backend=GPUBackend())
```

### "Structure function is NaN"
```julia
# Solution: Check for NaN inputs or remove them
sum(isnan.(u))  # Check NaNs
u_clean = u[.!isnan.(u), :]  # Remove

# Or let calculate_structure_function handle it
result = calculate_structure_function(x, u, bins)  # Skips NaN pairs
```

### "Serial computation is too slow"
```julia
# Solution: Use ThreadedBackend
backend = ThreadedBackend()  # Requires OhMyThreads

# Or try GPU
backend = GPUBackend()  # Requires KernelAbstractions
```

---

## Further Reading

After the examples, dive into the comprehensive docs:

- **Theory**: `docs/theory.md` — Physics and mathematics
- **Architecture**: `docs/architecture.md` — Internal design
- **Backends**: `docs/backends.md` — Detailed backend guide
- **Extensions**: `docs/extensions.md` — Lazy loading system

---

## Contributing Examples

To add a new example:

1. Create `examples/new_example.jl` with:
   - Clear docstring explaining what it does
   - Section headers for organization
   - Comments explaining key concepts
   - Print statements summarizing results
   - Links to relevant docs

2. Update this README with:
   - Brief description
   - Run instructions
   - Key concepts learned
   - Next steps

3. Test it runs without errors:
   ```bash
   julia examples/new_example.jl
   ```

---

## Questions?

- Check docstrings: `?calculate_structure_function`
- Search examples/ for similar workflows
- Read the relevant doc file (theory / backends / gpu)
- Open an issue on GitHub

---

**Happy computing! 🚀**

Last updated: v0.3.0
