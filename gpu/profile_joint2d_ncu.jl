#!/usr/bin/env julia
"""
    profile_joint2d_ncu.jl

Minimal joint-2D launches for Nsight Compute (`ncu`).

CUDA.jl headless recipe: `ncu julia gpu/profile_joint2d_ncu.jl` with
`--kernel-name-base demangled` and `--launch-skip` = `PREWARM`.
See `gpu/run_ncu_joint2d.sh` and CUDA.jl profiling docs.

Environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `COMPILE_CELLS` | `exact` | `exact` (NB2), `max` (4096), or integer |
| `PREWARM` | `3` | full launches before the final profiled launch |
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Printf: Printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges, joint2d_smem_max
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT

const _GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
_GPUExt === nothing && error("StructureFunctionsKernelAbstractionsExt not loaded — use julia --project=gpu")

function _dist_bins(n_dist::Int, ::Type{FT}) where {FT}
    return LogBinEdges(Vector{FT}(exp.(range(log(FT(1000)), log(FT(50000)); length = n_dist + 1))))
end

function _value_bins(route::AbstractString, n_val_inner::Int, ::Type{FT}) where {FT}
    if route == "general"
        inner = collect(FT, range(-1, 2; length = n_val_inner + 1))
        return vcat(FT(-Inf), inner, FT(Inf))
    end
    return InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val_inner + 1)))
end

function _resolve_compile_cells(nb2::Int)
    spec = get(ENV, "COMPILE_CELLS", "exact")
    spec == "exact" && return nothing
    spec == "max" && return joint2d_smem_max()
    return parse(Int, spec)
end

function main()
    CUDA.functional() || error("CUDA not functional")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    prewarm = parse(Int, get(ENV, "PREWARM", "3"))
    route = lowercase(get(ENV, "VALUE_ROUTE", "inflinear"))
    FT = Float32
    backend = CUDA.CUDABackend()

    dist = _dist_bins(n_dist, FT)
    value_bins = _value_bins(route, n_val_inner, FT)
    nb2 = (length(dist) - 1) * (length(value_bins) - 1)
    compile_kw = _resolve_compile_cells(nb2)

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    ws = if compile_kw === nothing
        SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :joint2d)
    else
        SFC.GPUSFWorkspace(
            backend, dist, value_bins;
            kind = :joint2d, joint2d_compile_cells = compile_kw,
        )
    end

    launch! = function ()
        SFC.reset_histogram!(ws)
        _GPUExt._launch_gpu_joint2d!(
            sft, backend, x, u, dist, value_bins;
            workspace = ws, synchronize = true,
        )
        return nothing
    end

    for _ in 1:prewarm
        launch!()
    end

    _GPUExt._joint2d_resolve_tiled_kernel!(
        ws, backend, ws.dist_bins, ws.val_plan, ws.joint2d_compile_cells,
    )
    dist_r = _GPUExt._joint2d_dist_route(ws.dist_bins)
    val_r = _GPUExt._joint2d_val_route(ws.val_plan)
    gpu_fn = ws.joint2d_kernel === nothing ? :unknown : nameof(ws.joint2d_kernel.f)
    msg = Printf.@sprintf(
        "ncu workload: N=%d n_dist=%d n_val=%d NB2=%d route=%s/%s compile_cells=%d prewarm=%d gpu_fn=%s",
        N, n_dist, length(value_bins) - 1, nb2, dist_r, val_r, ws.joint2d_compile_cells, prewarm, gpu_fn,
    )
    println(msg)
    flush(stdout)

    # Final launch — ncu --launch-skip PREWARM profiles this one (with demangled sf2d filter).
    launch!()
    return nothing
end

main()
