#!/usr/bin/env julia
"""
    benchmark_suite.jl

Maintained GPU benchmark entry point.

Run inside a GPU allocation for CUDA:

    julia --project=gpu gpu/benchmark_suite.jl

Useful environment variables:

- `BENCH_BACKEND=auto|cuda|kacpu` (default: `auto`)
- `N=20000`
- `BATCH=16`
- `N_DIST=32`
- `N_VAL=16`
- `REPEAT=3`
- `WARMUP=1`
"""

using ComputationalBackends: ComputationalBackends as CB
using CUDA: CUDA
using Dates: Dates
using JSON: JSON
using KernelAbstractions: KernelAbstractions as KA
using Random: Random
using Statistics: Statistics
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

const RESULT_DIR = joinpath(@__DIR__, "benchmark_results")

function _env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function _select_backend()
    mode = lowercase(get(ENV, "BENCH_BACKEND", "auto"))
    if mode == "cuda"
        CUDA.functional() || error("BENCH_BACKEND=cuda requested, but CUDA.functional() is false")
        return CUDA.CUDABackend(), "cuda"
    elseif mode == "kacpu"
        return KA.CPU(), "kacpu"
    elseif mode == "auto"
        if CUDA.functional()
            return CUDA.CUDABackend(), "cuda"
        end
        return KA.CPU(), "kacpu"
    end
    error("BENCH_BACKEND must be auto, cuda, or kacpu; got $mode")
end

function _stage(backend, a)
    return backend isa CUDA.CUDABackend ? CUDA.CuArray(a) : a
end

function _timed(label::String, backend, f; warmup::Int, repeat::Int)
    for _ in 1:warmup
        f()
    end
    gpu_sync!(backend)
    times = Float64[]
    for _ in 1:repeat
        t = @elapsed f()
        gpu_sync!(backend)
        push!(times, t)
    end
    return Dict(
        "label" => label,
        "seconds_median" => Statistics.median(times),
        "seconds_min" => minimum(times),
        "seconds_all" => times,
    )
end

function _ratio(a::Real, b::Real)
    return b == 0 ? NaN : Float64(a) / Float64(b)
end

function _explicit_aux_loop_shared_positions(sft, backend, x, u, distance_bins)
    @views for b in axes(u, 3)
        SFC.gpu_calculate_structure_function(
            sft, backend, x, u[:, :, b], distance_bins)
    end
    return nothing
end

function _explicit_aux_loop_varying_positions(sft, backend, x, u, distance_bins)
    @views for b in axes(u, 3)
        SFC.gpu_calculate_structure_function(
            sft, backend, x[:, :, b], u[:, :, b], distance_bins)
    end
    return nothing
end

function main()
    Random.seed!(42)
    backend, backend_name = _select_backend()
    FT = Float32
    N = _env_int("N", backend_name == "cuda" ? 20_000 : 512)
    B = _env_int("BATCH", backend_name == "cuda" ? 16 : 4)
    n_dist = _env_int("N_DIST", 32)
    n_val = _env_int("N_VAL", 16)
    repeat = _env_int("REPEAT", 3)
    warmup = _env_int("WARMUP", 1)

    sft = SFT.L2SFType()
    dist_bins = LinearBinEdges(range(FT(0), FT(1.5); length = n_dist + 1))
    value_bins = LinearBinEdges(range(FT(-1), FT(2); length = n_val + 1))

    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    xd = _stage(backend, x)
    ud = _stage(backend, u)
    x3 = rand(FT, 3, N)
    u3 = rand(FT, 3, N)
    xd3 = _stage(backend, x3)
    ud3 = _stage(backend, u3)

    rows = Dict{String, Any}[]

    ws1d = SFC.GPUSFWorkspace(backend, dist_bins)
    push!(rows, _timed("sf1d_fresh", backend, () -> begin
        SFC.gpu_calculate_structure_function(sft, backend, xd, ud, dist_bins)
    end; warmup = warmup, repeat = repeat))
    push!(rows, _timed("sf1d_workspace", backend, () -> begin
        SFC.gpu_calculate_structure_function(
            sft, backend, xd, ud, dist_bins; workspace = ws1d,
        )
    end; warmup = warmup, repeat = repeat))
    push!(rows, _timed("sf1d_3d_workspace", backend, () -> begin
        SFC.gpu_calculate_structure_function(
            sft, backend, xd3, ud3, dist_bins; workspace = ws1d,
        )
    end; warmup = warmup, repeat = repeat))

    ws_joint = SFC.GPUSFWorkspace(backend, dist_bins, value_bins; kind = :joint2d)
    push!(rows, _timed("joint2d_workspace", backend, () -> begin
        SFC.gpu_calculate_structure_function_2d(sft, backend, xd, ud, dist_bins, value_bins; workspace = ws_joint)
    end; warmup = warmup, repeat = repeat))

    ws_sp2d = SFC.GPUSFWorkspace(backend, dist_bins, value_bins; kind = :single_pass_2d)
    push!(rows, _timed("sp2d_workspace", backend, () -> begin
        SFC.gpu_calculate_structure_functions_single_pass_2d(backend, xd, ud, dist_bins, value_bins; workspace = ws_sp2d)
    end; warmup = warmup, repeat = repeat))

    u_shared = rand(FT, 2, N, B)
    xd_shared = _stage(backend, x)
    ud_shared = _stage(backend, u_shared)
    push!(rows, _timed("aux_shared_positions_sf1d", backend, () -> begin
        SFC.calculate_structure_function(
            sft, xd_shared, ud_shared, dist_bins;
            backend = CB.GPUBackend(backend), verbose = false,
        )
    end; warmup = warmup, repeat = repeat))
    push!(rows, _timed("aux_shared_positions_explicit_loop", backend, () -> begin
        _explicit_aux_loop_shared_positions(sft, backend, xd_shared, ud_shared, dist_bins)
    end; warmup = warmup, repeat = repeat))

    x_varying = rand(FT, 2, N, B)
    u_varying = rand(FT, 2, N, B)
    xd_varying = _stage(backend, x_varying)
    ud_varying = _stage(backend, u_varying)
    push!(rows, _timed("aux_varying_positions_sf1d", backend, () -> begin
        SFC.calculate_structure_function(
            sft, xd_varying, ud_varying, dist_bins;
            backend = CB.GPUBackend(backend), verbose = false,
        )
    end; warmup = warmup, repeat = repeat))
    push!(rows, _timed("aux_varying_positions_explicit_loop", backend, () -> begin
        _explicit_aux_loop_varying_positions(sft, backend, xd_varying, ud_varying, dist_bins)
    end; warmup = warmup, repeat = repeat))

    by_label = Dict(row["label"] => row["seconds_median"] for row in rows)
    gates = Dict(
        "workspace_speedup" => _ratio(by_label["sf1d_fresh"], by_label["sf1d_workspace"]),
        "sp2d_vs_6x_joint2d" => _ratio(6 * by_label["joint2d_workspace"], by_label["sp2d_workspace"]),
        "shared_aux_vs_explicit_loop" => _ratio(
            by_label["aux_shared_positions_explicit_loop"],
            by_label["aux_shared_positions_sf1d"],
        ),
        "varying_aux_vs_explicit_loop" => _ratio(
            by_label["aux_varying_positions_explicit_loop"],
            by_label["aux_varying_positions_sf1d"],
        ),
    )

    result = Dict(
        "timestamp" => string(Dates.now()),
        "backend" => backend_name,
        "device" => backend isa CUDA.CUDABackend ? CUDA.name(CUDA.device()) : string(typeof(backend)),
        "N" => N,
        "batch" => B,
        "n_dist" => n_dist,
        "n_val" => n_val,
        "warmup" => warmup,
        "repeat" => repeat,
        "rows" => rows,
        "gates" => gates,
    )

    mkpath(RESULT_DIR)
    latest = joinpath(RESULT_DIR, "benchmark_suite_latest.json")
    stamped = joinpath(RESULT_DIR, "benchmark_suite_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).json")
    open(latest, "w") do io
        JSON.print(io, result, 2)
    end
    open(stamped, "w") do io
        JSON.print(io, result, 2)
    end

    println("backend: ", backend_name)
    println("device: ", result["device"])
    println("N=$N batch=$B n_dist=$n_dist n_val=$n_val repeat=$repeat")
    for row in rows
        println(rpad(row["label"], 40), round(row["seconds_median"]; digits = 6), " s")
    end
    println("workspace speedup: ", round(gates["workspace_speedup"]; digits = 3), "x")
    println("6x joint2d / sp2d: ", round(gates["sp2d_vs_6x_joint2d"]; digits = 3), "x")
    println("shared aux explicit/fused: ", round(gates["shared_aux_vs_explicit_loop"]; digits = 3), "x")
    println("varying aux explicit/fused: ", round(gates["varying_aux_vs_explicit_loop"]; digits = 3), "x")
    println("wrote: ", latest)

    SFC.release!(ws1d)
    SFC.release!(ws_joint)
    SFC.release!(ws_sp2d)
    return result
end

Base.invokelatest(main)
