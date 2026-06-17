"""
    benchmark_scaling_helpers.jl

Reusable timing helpers for GPU workspace, slice-batch, and scaling collectors.
Included by `benchmark_slices.jl`, `benchmark_workspace.jl`, and `collect_benchmark_assets.jl`.

Expects `CUDA` to be loaded by the including script when using `CUDA.CUDABackend()`.
"""

using StructureFunctions: Calculations as SFC
using Statistics: Statistics

"""
    gpu_sync!(backend)

Synchronize after GPU kernel launches when using CUDA.
"""
function gpu_sync!(backend)
    if backend isa CUDA.CUDABackend
        CUDA.synchronize()
    end
    return nothing
end

"""
    run_timed_gpu(f, backend; warmup=1) -> Float64

Warm up `warmup` times, then time one call; synchronize CUDA when applicable.
"""
function run_timed_gpu(f, backend; warmup::Int = 1)
    for _ in 1:warmup
        f()
    end
    gpu_sync!(backend)
    t = @elapsed f()
    gpu_sync!(backend)
    return t
end

"""
    bench_cpu_serial_sf(x_tup, u_tup, bins, sft; warmup=1) -> Float64

Time one **serial** CPU structure-function call (GPU doc assets — always 1 logical CPU worker).
Thread scaling is measured separately in `benchmark/benchmark_scaling.jl`.
"""
function bench_cpu_serial_sf(x_tup, u_tup, bins, sft; warmup::Int = 1)
    for _ in 1:warmup
        SFC.calculate_structure_function(
            sft, x_tup, u_tup, bins;
            backend = SFC.SerialBackend(), verbose = false, show_progress = false,
        )
    end
    return @elapsed SFC.calculate_structure_function(
        sft, x_tup, u_tup, bins;
        backend = SFC.SerialBackend(), verbose = false, show_progress = false,
    )
end

"""
    bench_gpu_sf_with_workspace(backend, x_dev, u_dev, bins, sft, ws; warmup=2, repeat=3) -> Float64

Time GPU calls reusing `GPUSFWorkspace`. Runs `warmup` untimed launches, then `repeat`
timed launches; returns the **median** (sub-ms GPU times are noisy with `repeat=1`).
"""
function bench_gpu_sf_with_workspace(
    backend, x_dev, u_dev, bins, sft, ws; warmup::Int = 2, repeat::Int = 3,
)
    for _ in 1:warmup
        SFC.gpu_calculate_structure_function(
            sft, backend, x_dev, u_dev, bins;
            return_sums_and_counts = true, workspace = ws,
        )
    end
    gpu_sync!(backend)
    times = Float64[]
    for _ in 1:repeat
        t = @elapsed SFC.gpu_calculate_structure_function(
            sft, backend, x_dev, u_dev, bins;
            return_sums_and_counts = true, workspace = ws,
        )
        gpu_sync!(backend)
        push!(times, t)
    end
    return Statistics.median(times)
end

"""
    bench_gpu_sf_fresh(backend, x_dev, u_dev, bins, sft; warmup=1) -> Float64

Time one GPU call without workspace (fresh device histogram alloc each call).
"""
function bench_gpu_sf_fresh(backend, x_dev, u_dev, bins, sft; warmup::Int = 1)
    for _ in 1:warmup
        SFC.gpu_calculate_structure_function(
            sft, backend, x_dev, u_dev, bins;
            return_sums_and_counts = true,
        )
    end
    gpu_sync!(backend)
    t = @elapsed SFC.gpu_calculate_structure_function(
        sft, backend, x_dev, u_dev, bins;
        return_sums_and_counts = true,
    )
    gpu_sync!(backend)
    return t
end

"""
    bench_naive_slice_loop!(backend, x_host, u_host, bins, sft, sums, counts; T, warmup=1)

Per-slice host upload + fresh GPU alloc each time step.
"""
function bench_naive_slice_loop!(
    backend, x_host, u_host, bins, sft, sums, counts; T::Int, warmup::Int = 1,
)
    function run!()
        for t in 1:T
            res = SFC.gpu_calculate_structure_function(
                sft, backend, x_host[:, :, t], u_host[:, :, t], bins;
                return_sums_and_counts = true,
            )
            sums[:, t] .= res.sums
            counts[:, t] .= res.counts
        end
    end
    return run_timed_gpu(run!, backend; warmup = warmup)
end

"""
    bench_slice_driver!(backend, x_batch, u_batch, bins, sft, sums, counts, ws; warmup=1)

Batch slice driver API (`gpu_calculate_structure_function_slices!`).
"""
function bench_slice_driver!(
    backend, x_batch, u_batch, bins, sft, sums, counts, ws; warmup::Int = 1,
)
    function run!()
        SFC.gpu_calculate_structure_function_slices!(
            sums, counts, sft, backend, x_batch, u_batch, bins; workspace = ws,
        )
    end
    return run_timed_gpu(run!, backend; warmup = warmup)
end

"""
    bench_cpu_serial_slice_loop!(x_batch, u_batch, bins, sft, sums, counts; T, warmup=1)

Serial CPU per-slice loop (same 1-worker policy as [`bench_cpu_serial_sf`](@ref)).
"""
function bench_cpu_serial_slice_loop!(x_batch, u_batch, bins, sft, sums, counts; T::Int, warmup::Int = 1)
    function run!()
        for t in 1:T
            x_t = (x_batch[1, :, t], x_batch[2, :, t], x_batch[3, :, t])
            u_t = (u_batch[1, :, t], u_batch[2, :, t], u_batch[3, :, t])
            res = SFC.calculate_structure_function(
                sft, x_t, u_t, bins;
                backend = SFC.SerialBackend(), verbose = false, show_progress = false,
                return_sums_and_counts = true,
            )
            sums[:, t] .= res.sums
            counts[:, t] .= res.counts
        end
    end
    for _ in 1:warmup
        run!()
    end
    return @elapsed run!()
end

"""
    stage_device_arrays(backend, x_host, u_host, ::Type{FT})

Upload host `(3, N)` CPU arrays to device when `backend` is CUDA.

Use `CUDA.CuArray{FT}(...)` — not `CUDA.cu(...)` — so Float64 host data stays
Float64 on device (on many setups `CUDA.cu` silently promotes to the device default,
often Float32).
"""
function stage_device_arrays(backend, x_host, u_host, ::Type{FT}) where {FT}
    size(x_host, 1) == 3 || throw(ArgumentError("x must have shape (3, N); got $(size(x_host))"))
    size(u_host) == size(x_host) ||
        throw(ArgumentError("u must match x shape $(size(x_host)); got $(size(u_host))"))
    eltype(x_host) == FT ||
        throw(ArgumentError("x eltype $(eltype(x_host)) != requested $FT"))
    eltype(u_host) == FT ||
        throw(ArgumentError("u eltype $(eltype(u_host)) != requested $FT"))
    if backend isa CUDA.CUDABackend
        return CUDA.CuArray{FT}(x_host), CUDA.CuArray{FT}(u_host)
    end
    return x_host, u_host
end

"""
    stage_device_batch(backend, x_host, u_host, ::Type{FT})

Upload host `(3, N, T)` batch to device when using CUDA (same `CuArray{FT}` rule).
"""
function stage_device_batch(backend, x_host, u_host, ::Type{FT}) where {FT}
    size(x_host, 1) == 3 || throw(ArgumentError("x batch must have shape (3, N, T); got $(size(x_host))"))
    size(u_host) == size(x_host) ||
        throw(ArgumentError("u batch must match x shape $(size(x_host)); got $(size(u_host))"))
    eltype(x_host) == FT ||
        throw(ArgumentError("x batch eltype $(eltype(x_host)) != requested $FT"))
    eltype(u_host) == FT ||
        throw(ArgumentError("u batch eltype $(eltype(u_host)) != requested $FT"))
    if backend isa CUDA.CUDABackend
        return CUDA.CuArray{FT}(x_host), CUDA.CuArray{FT}(u_host)
    end
    return x_host, u_host
end
