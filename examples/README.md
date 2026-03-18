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

**Best for**: Large datasets (>500M points), with GPU hardware

```bash
# Requires CUDA.jl + KernelAbstractions.jl
julia --project -e 'using Pkg; Pkg.add(["CUDA", "KernelAbstractions"])'

# Run example
julia examples/gpu_acceleration.jl
```

**What it does**:
- Generates 1 billion points (or 10M if GPU unavailable)
- Uses GPUBackend for computation
- Measures memory efficiency of Float32 vs Float64
- Estimates speedup

**Expected results**:
- NVIDIA A100: 20–50x faster than single CPU
- RTX 4090: 10–30x faster than CPU
- AMD MI250X: 15–80x faster than CPU

**Key insights**:
- Float32 sufficient for turbulence (saves 50% memory)
- GPU shines for >1B points
- Kernel compilation amortized over many calls

**Learn next**: `docs/backends.md#gpubackend`, `docs/real_data.md#performance-tips`

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

### 5. Real Climate Data (`real_data_climate.jl`)

**Best for**: Practical atmospheric/ocean applications

```bash
# Optional: install NetCDF support
julia -e 'using Pkg; Pkg.add("NetCDF")'

# Run with synthetic data (generates demo)
julia examples/real_data_climate.jl
```

**What it does**:
- Simulates realistic atmospheric turbulence data
- Handles missing data (NaNs) and outliers
- Constructs 3D coordinate systems
- Performs K41 scaling analysis
- Multi-time-step statistics

**Key workflows**:
1. Data loading (NetCDF, HDF5, CSV)
2. Preprocessing (unit conversion, detrending)
3. Structure function calculation
4. K41 validation
5. Temporal statistics

**Real data sources**:
- SOCRATES campaign: https://data.eol.ucar.edu/
- ERA5 reanalysis: https://cds.climate.copernicus.eu/
- MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
- WRF simulations: Custom NetCDF output

**Learn next**: `docs/real_data.md`

---

## Advanced Workflows

### Combining Backends

**Scenario**: Process each file with GPU, aggregate across time

```julia
using StructureFunctions, DistributedArrays

# Load time series in parallel
for t in 1:n_time_steps
    # Each worker computes one time step
    result_t = @distributed (+) for i in workers()
        data_i = load_time_slice(t, i)
        x, u = construct_coordinates(data_i)
        calculate_structure_function(x, u, bins; backend=GPUBackend())
    end
end
```

### Batch Processing Out-of-Core

**Scenario**: Dataset too large for memory (terabytes)

```julia
# Process in streaming chunks
all_sums = zeros(n_bins)
all_counts = zeros(Int, n_bins)

for chunk in read_chunks("huge_data.zarr")
    x, u = extract_data(chunk)
    result = calculate_structure_function(x, u, bins; backend=AutoBackend())
    all_sums .+= result.sums
    all_counts .+= result.counts
end

# Normalize at end
final_sf = all_sums ./ all_counts
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
for script in simple_2d threaded_calculation gpu_acceleration distributed_parallel real_data_climate; do
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
| real_data_climate | 50K pts | ~0.5 sec | Serial |

*Actual times vary by hardware. Use `@time` to measure.*

---

## Choosing an Example to Start With

### I want to...

- **Learn the basics** → `simple_2d.jl`
- **Speed up my computation** → `threaded_calculation.jl`
- **Use my GPU** → `gpu_acceleration.jl`
- **Scale to a supercomputer** → `distributed_parallel.jl`
- **Analyze real atmospheric data** → `real_data_climate.jl`
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
- **Real Data**: `docs/real_data.md` — File I/O and workflows

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
- Read the relevant doc file (theory / backends / real_data)
- Open an issue on GitHub

---

**Happy computing! 🚀**

Last updated: v0.3.0
