# Profiling and Optimization Open Issues Log

This document tracks identified performance bottlenecks, memory allocation issues, proposed solutions, and potential blockers found during profiling in `StructureFunctions.jl`.

---

## 1. Threaded Backend: Chunk-Local Allocations in single-pass pair loops

### Problem Description
During memory allocation profiling (captured in `allocs_threaded.txt`), the `ThreadedBackend` (implemented in `StructureFunctionsOhMyThreadsExt.jl`) was found to perform chunk-local array allocations inside the `tmapreduce` reduction block:
```julia
local_sums = zeros(OT, 8, n_bins)
local_counts = zeros(Int64, 8, n_bins)
```
These allocations occur once per parallel chunk/task. With 24 threads/tasks, this results in exactly 48 allocations ($\approx 24\text{ KB}$ total). While this is a minor $O(\text{nthreads})$ overhead (and is allocation-free *inside* the main pairwise loop), it prevents the threaded backend from achieving absolute zero allocations. 

Standard thread-indexed pre-allocated buffers (like `thread_sums[:, :, threadid()]`) cannot be used because Julia tasks are non-sticky and migrate between threads, which causes data races and glibc heap corruption.

### Proposed Solution
We can pre-allocate chunk-local buffers and map them 1-to-1 to tasks by using the chunk index instead of `threadid()`:
1. Chunk the loop using `enumerate(_triangle_outer_chunks(1:n_points, n_tasks))`, which yields `(chunk_idx, chunk)` for each task.
2. Index into a pre-allocated vector of matrices (`Vector{Matrix{OT}}`) or a 3D array (`AbstractArray{OT, 3}`) using the unique `chunk_idx`:
   ```julia
   local_sums = view(thread_sums, :, :, chunk_idx)
   local_counts = view(thread_counts, :, :, chunk_idx)
   ```
3. Zero the buffer values before accumulation using `fill!(local_sums, 0)` and `fill!(local_counts, 0)`.
4. This ensures that each chunk gets a unique, thread-safe pre-allocated slice, enabling 100% thread-safe execution with absolute zero heap allocations.

### Blockers / Design Considerations
* **Wrapper View Allocations**: Taking a `view` of a 3D array can sometimes allocate a small slice wrapper object (SubArray) on the heap if it escapes or fails to compile down. Using a `Vector{Matrix{OT}}` is guaranteed to be 100% allocation-free but requires the caller to preallocate a vector of matrices instead of a standard 3D array.
* **API Extension Synchronization**: We need to update the keyword argument parsing in `Calculations.jl` and verify that `thread_sums`/`thread_counts` are cleanly passed down via `kwargs` to `_dispatch_single_pass!` in all backend extensions.

---

## 2. Redundant normalization of unit vector in HelperFunctions.n̂ for 2D inputs

### Problem Description
The CPU profile flat report (`cpu_serial.txt`) shows significant execution time spent inside `normalize` (~2804 snapshots) and `n̂` (~2428 snapshots). 
In `src/HelperFunctions.jl`, the `n̂(r_hat)` function is defined as:
```julia
function n̂(r_hat::AbstractVector{FT}) where {FT}
    ND::Int = length(r_hat)
    if ND == 2
        return LA.normalize(SA.SVector{2, FT}(r_hat[2], -r_hat[1]))
    ...
```
However, the input parameter `r_hat` is the longitudinal unit vector from `r̂(x1, x2)`, which is *already* normalized to unit length ($1.0$). 
For 2D vectors, rotating by 90 degrees (`[y, -x]`) preserves the vector norm exactly:
$$\|[y, -x]\|_2^2 = y^2 + (-x)^2 = x^2 + y^2 = \|[x, y]\|_2^2 = 1.0$$
Calling `LA.normalize` on it is completely redundant and wastes CPU cycles by computing a square root and performing division.

### Proposed Solution
In 2D, return the rotated `SA.SVector` directly without calling `LA.normalize`:
```julia
    if ND == 2
        return SA.SVector{2, FT}(r_hat[2], -r_hat[1])
```
This is consistent with the fallback signature for tuples on line 126 (`n̂(r_hat::NTuple{2, T}) = SA.SVector(r_hat[2], -r_hat[1])`), which does not perform redundant normalization.

### Blockers / Design Considerations
None. This is a safe and immediate performance optimization.

---

## 3. Impact of @fastmath and Branch-free Optimizations in the Pairwise Loop

### Problem Description
To investigate potential arithmetic and loop vectorization optimizations in the main pairwise loop body, we compared four different loop designs on a dataset of $N = 3000$ points and $50$ linear bins:
1. **Branching + Standard Math**: Uses standard control-flow boundaries (`if 1 <= bin_idx <= n_bins`).
2. **Branching + `@fastmath`**: Standard control-flow with `@fastmath`.
3. **Branch-free + Standard Math**: Bypasses the `if` check by clamping the bin index (`clamp(bin_idx, 1, n_bins + 2)`) to write to a padded array buffer.
4. **Branch-free + `@fastmath`**: Padded array buffer with `@fastmath`.

### Benchmark Results
* **1. Branching + Standard Math**: $41.89\text{ ms}$
* **2. Branching + `@fastmath`**: $41.84\text{ ms}$ ($\approx 0.1\%$ speedup)
* **3. Branch-free + Standard Math**: $37.42\text{ ms}$ ($\approx 10.7\%$ speedup)
* **4. Branch-free + `@fastmath`**: $37.42\text{ ms}$ ($\approx 0.0\%$ speedup over branch-free standard math)

### Key Findings and Compiler Analysis
1. **Branch-free yields real speedups**: Eliminating the control-flow `if` check and replacing it with a branch-free `clamp` (which compiles to branch-free conditional select/move instructions) eliminates CPU pipeline branch mispredictions, yielding a consistent **$\approx 10.7\%$ performance improvement**.
2. **`@fastmath` still has zero effect**: Even when the code is completely branch-free, applying `@fastmath` provides absolutely no additional speedup.
3. **Scatter Hazards block SIMD Vectorization**: The reason `@fastmath` fails to speed up the loop is that LLVM is blocked from auto-vectorizing the loop because of the dynamic binned reduction:
   ```julia
   @inbounds sums[1, clamped_idx] += du_L2 + du_T2
   ```
   Because `clamped_idx` is dynamically computed at runtime per pair (based on distance `r`), LLVM cannot prove that different lanes in a hypothetical SIMD register will not write to the same `clamped_idx` at the same time. This loop-carried data hazard (scatter hazard) forces the compiler to execute the updates sequentially, preventing vectorization. Without vectorization, `@fastmath`'s arithmetic optimizations (reassociation, etc.) have no impact.

### Recommendations
* **Do not use `@fastmath`** in the pairwise loop, as it is ineffective due to the scatter reduction hazard.
* **Adopt Branch-free Clamping**: Use padded array buffers (`n_bins + 2`) to safely accumulate underflow/overflow values and use `clamp(bin_idx, 1, n_bins + 2)` to avoid branch misprediction overhead.
* **Prioritize Algorithmic Complexity**: Focus on spatial pruning (Cell Lists) to reduce the calculation from $O(N^2)$ to $O(N)$, which is the most impactful way to scale the code.

---

## 4. Memory Layout Optimization: Column-Major Matrix vs. Contiguous AoS vs. Struct-of-Arrays (SoA)

### Problem Description
Currently, coordinates and velocity fields are stored and indexed as 2D `Matrix{Float64}` structures of size `(2, N)`, from which SVectors are constructed on the fly. We investigated whether reorganizing memory layouts could yield performance gains by improving CPU prefetching and cache hit rates in the tight pairwise loop, and whether we could achieve these gains with **zero allocations**.

We benchmarked four memory layouts on a dataset of $N = 3000$ points with $n_{\text{bins}} = 50$ (using the branch-free `clamp` optimization):
1. **Column-Major Matrix (Current)**: Coordinates read dynamically from columns of `(2, N)` matrix to construct `SVector`s.
2. **Vector of SVectors (Contiguous AoS)**: Contiguous vector of pre-allocated 16-byte SVectors (`x[j]`).
3. **SoA via Row Views on `(2, N)`**: Slices taken via views `@view x[1, :]`. This creates stride-2 views (non-contiguous memory rows due to column-major layout).
4. **SoA via Column Views on Transposed `(N, 2)`**: Slices taken via views `@view x[:, 1]`. Since columns are contiguous in column-major layout, this creates stride-1 views.
5. **SoA via Contiguous Copies from `(2, N)`**: Slices copied into new vectors `x[1, :]`.

### Benchmark Results
* **1. Column-Major Matrix (Current)**: $36.34\text{ ms}$
* **2. Vector of SVectors (AoS)**: $38.51\text{ ms}$ ($\approx 6\%$ slower than current)
* **3. SoA via Row Views `(2, N)`**: $34.69\text{ ms}$ (0 allocations, but slower than copies due to stride-2 memory lookups)
* **4. SoA via Contiguous Copies**: $31.32\text{ ms}$ (12 allocations: $94\text{ KiB}$)
* **5. SoA via Column Views `(N, 2)` (Transposed)**: **$31.13\text{ ms}$ (0 allocations)**

### Key Findings and Compiler Analysis
1. **Contiguous Column Views**: In Julia's column-major layout, slicing a matrix column `x[:, 1]` creates a `SubArray` view with **stride = 1** (elements are physically adjacent in memory). This achieves the exact same computational efficiency as contiguous copied vectors while performing **exactly 0 heap allocations**.
2. **The Stride Penalty on Row Views**: Taking a row view `@view x[1, :]` on a `(2, N)` matrix results in a stride of 2. Inside the hot inner loop, stride-2 indexing requires address-calculation math on every read and degrades CPU cache prefetching, neutralizing the benefits of register caching.
3. **Copy vs. View Trade-off**: If input matrices must stay in the current `(2, N)` layout, copying rows to contiguous 1D vectors is faster than using row views, despite the tiny one-time allocation cost of $94\text{ KiB}$ for $N=3000$.
4. **Clean SVector Math is Allocation-Free**: Because `SVector` is a stack-allocated primitive value type, constructing it on-the-fly in the loop (e.g., `x_j = SVector{2, FT}(xs[j], ys[j])` or directly `x_j = SVector{2, FT}(x_mat[j, 1], x_mat[j, 2])`) has **exactly 0 heap allocations**. The compiler compiles this down to the same register instructions as scalar math, resulting in a negligible performance cost ($\approx 6\%$ difference) while fully preserving clean vector abstractions, projections, and `HelperFunctions` (`n̂`, `r̂`, etc.).

### Recommendations
* **Enforce and Document the `(N, D)` Input Layout**: Clearly specify in the package API that coordinates and velocities should be passed as matrices of shape `(N, D)` (where row `i` represents point `i` and columns are dimensions).
* **Retain SVector Abstractions**: Do not decompose calculations into manual scalar algebra. Instead, construct `SVector`s on the fly from the contiguous inputs:
  ```julia
  # Inside the loop:
  x_i = SVector{2, FT}(x[i, 1], x[i, 2])
  # ...
  x_j = SVector{2, FT}(x[j, 1], x[j, 2])
  ```
  This is 100% allocation-free and compiles to highly optimized vector math while keeping the codebase clean, readable, and fully compatible with existing helper projection methods.
* **No Runtime Shims or Magic Transpositions**: Do not perform automatic runtime transpositions or copies inside the package. Rely on clear API constraints and let the caller manage their matrix layout.
