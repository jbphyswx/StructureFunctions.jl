# Profiling open issues

## Threaded backend: per-task accumulator allocation

`ThreadedBackend` (`ext/StructureFunctionsOhMyThreadsExt.jl`) runs each pair loop through
`_greedy_reduce`, in which every task makes its accumulator once with `init()`, for example

```julia
() -> ((zeros(OT, SFC.SINGLE_PASS_N, n_bins), zeros(CT, SFC.SINGLE_PASS_N, n_bins)), nothing)
```

The allocation count is `O(nthreads)` per call and the pair loop itself allocates nothing, but the
threaded backend is not allocation-free.

Thread-indexed preallocated buffers (`thread_sums[:, :, threadid()]`) are not an option: Julia tasks
migrate between threads, so indexing by `threadid()` races.

Preallocated per-task buffers need a new argument threaded through the threaded entries; no entry
takes chunk buffers today. A `Vector{Matrix{OT}}` is allocation-free to index, whereas a `view` of a
3D array can allocate a `SubArray` wrapper when it escapes.
