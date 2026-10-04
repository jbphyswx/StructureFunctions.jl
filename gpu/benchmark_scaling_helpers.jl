"""
    benchmark_scaling_helpers.jl

Timing helpers of `benchmark_slices.jl`, `benchmark_workspace.jl` and `collect_benchmark_assets.jl`, which load
`CUDA` before including this file.
"""

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: Calculations as SFC
using StructureFunctions: StructureFunctionSumsAndCounts

"""
    gpu_sync!(backend)

Wait for the work queued on a CUDA `backend`.
"""
function gpu_sync!(backend)
    if backend isa CUDA.CUDABackend
        CUDA.synchronize()
    end
    return nothing
end

"""
    run_timed_gpu(f, backend; repeat=5) -> Float64

Seconds of the fastest of `repeat` synchronized calls of `f`, after one untimed call and half a second of calls that
bring the device to its working clock.
"""
function run_timed_gpu(f, backend; repeat::Int = 5)
    f()
    gpu_sync!(backend)
    t0 = time()
    while time() - t0 < 0.5
        f()
        gpu_sync!(backend)
    end
    return minimum(1:repeat) do _
        t = time_ns()
        f()
        gpu_sync!(backend)
        (time_ns() - t) / 1e9
    end
end

"""
    bench_cpu_serial_sf(x_arr, u_arr, bins, sft; repeat=7) -> Float64

Seconds of the fastest of `repeat` serial CPU calls, after one untimed call.
"""
function bench_cpu_serial_sf(x_arr, u_arr, bins, sft; repeat::Int = 7)
    f() = SFC.calculate_structure_function(sft, x_arr, u_arr, bins, UInt32, StructureFunctionSumsAndCounts;
                                           backend = CB.SerialBackend())
    f()
    return minimum(_ -> @elapsed(f()), 1:repeat)
end

"""
    bench_gpu_sf_with_workspace(backend, x_dev, u_dev, bins, sft, ws) -> Float64

Seconds per public GPU call reusing the `GPUSFWorkspace` `ws` ([`run_timed_gpu`](@ref)).
"""
bench_gpu_sf_with_workspace(backend, x_dev, u_dev, bins, sft, ws) =
    run_timed_gpu(() -> SFC.calculate_structure_function(sft, x_dev, u_dev, bins, UInt32, StructureFunctionSumsAndCounts;
                                                         backend = CB.GPUBackend(backend), workspace = ws), backend)

"""
    bench_gpu_sf_fresh(backend, x_dev, u_dev, bins, sft) -> Float64

Seconds per public GPU call without a workspace ([`run_timed_gpu`](@ref)).
"""
bench_gpu_sf_fresh(backend, x_dev, u_dev, bins, sft) =
    run_timed_gpu(() -> SFC.calculate_structure_function(sft, x_dev, u_dev, bins, UInt32, StructureFunctionSumsAndCounts;
                                                         backend = CB.GPUBackend(backend)), backend)

"""
    bench_naive_slice_loop!(backend, x_host, u_host, bins, sft, sums, counts; T)

Seconds for one public GPU call per time step, each uploading its slice and copying its result back.
"""
function bench_naive_slice_loop!(backend, x_host, u_host, bins, sft, sums, counts; T::Int)
    function run!()
        for t in 1:T
            res = SFC.calculate_structure_function(sft, x_host[:, :, t], u_host[:, :, t], bins, UInt32,
                                                   StructureFunctionSumsAndCounts; backend = CB.GPUBackend(backend))
            sums[:, t] .= Array(res.sums)
            counts[:, t] .= Array(res.counts)
        end
    end
    return run_timed_gpu(run!, backend)
end

"""
    bench_slice_driver!(backend, x_batch, u_batch, bins, sft, sums, counts, ws)

Seconds for one public batch call (`calculate_structure_function_batch!`) into the device buffers `sums`, `counts`,
which hold one call's histogram afterwards.
"""
function bench_slice_driver!(backend, x_batch, u_batch, bins, sft, sums, counts, ws)
    function run!()
        fill!(sums, 0)
        fill!(counts, 0)
        SFC.calculate_structure_function_batch!(sums, counts, sft, x_batch, u_batch, bins;
                                                backend = CB.GPUBackend(backend), workspace = ws)
    end
    return run_timed_gpu(run!, backend)
end

"""
    bench_cpu_serial_batch!(x_batch, u_batch, bins, sft, sums, counts; repeat=7) -> Float64

Seconds of the fastest of `repeat` serial CPU batch calls (`calculate_structure_function_batch!`) into `sums`,
`counts`, after one untimed call; the buffers hold one call's histogram afterwards.
"""
function bench_cpu_serial_batch!(x_batch, u_batch, bins, sft, sums, counts; repeat::Int = 7)
    function run!()
        fill!(sums, 0)
        fill!(counts, 0)
        SFC.calculate_structure_function_batch!(sums, counts, sft, x_batch, u_batch, bins;
                                                backend = CB.SerialBackend())
    end
    run!()
    return minimum(_ -> @elapsed(run!()), 1:repeat)
end

"""
    stage_device_arrays(backend, x_host, u_host, ::Type{FT})

The `(3, N)` host arrays on the device of a CUDA `backend`, in precision `FT`; other backends take them as they are.
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

The `(3, N, T)` host batch on the device of a CUDA `backend`, in precision `FT`; other backends take it as it is.
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
