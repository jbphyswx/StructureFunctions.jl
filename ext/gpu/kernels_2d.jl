# Tiled128 joint 2D SF histogram kernels (distance × value, typed digitize routes).
# Included from StructureFunctionsKernelAbstractionsExt.jl — block-local flat histogram in `@localmem`.
#
# Compile-time width `compile_cells` is chosen per [`GPUSFWorkspace`](@ref) (default exact
# `n_dist × n_val`; optional override via `joint2d_compile_cells`). Max-width kernels
# (`compile_cells == SF_GPU_MAX_2D_HIST`) are generated at load for each
# `(dist_route, val_route)` pair; other widths are `@eval`'d on first use.

const _JOINT2D_KERNEL_REGISTRY = Dict{Tuple{Symbol, Symbol, Int, Int}, Any}()
const _JOINT2D_DIST_ROUTES = (:linear, :log, :general)
const _JOINT2D_VAL_ROUTES = (:general, :linear, :inflinear, :log_linear)

"""Symbol name for a tiled joint kernel at `(dist_route, val_route, compile_cells, W)`."""
function _joint2d_kernel_fname(dist_route::Symbol, val_route::Symbol, compile_cells::Int, W::Int)
    base = Symbol(:_sf2d_kernel_tiled128_, dist_route, :_, val_route, :_w, W, :_u32)
    if compile_cells == SF_GPU_MAX_2D_HIST
        return Symbol(base, :!)
    end
    return Symbol(base, :_c, compile_cells, :_!)
end

"""Stage `W` components of one point into a tile buffer, unrolled at codegen."""
_joint2d_stage(dest::Symbol, src::Symbol, gidx::Symbol, W::Int) =
    Expr(:block, (:($(dest)[$((c - 1) * SF_GPU_TILE) + k] = $(src)[$c, $(gidx)]) for c in 1:W)...)

"""Load one `W`-component point out of a tile buffer."""
_joint2d_load(buf::Symbol, idx, W::Int) =
    :(SA.SVector{$W, FT}($((:($(buf)[$((c - 1) * SF_GPU_TILE) + $(idx)]) for c in 1:W)...)))

"""Cooperative grid-stride zero of block-local joint histogram (uses runtime `NB2`)."""
function _joint2d_cooperative_zero_body()
    return quote
    lid = @index(Local, Linear)
        b = lid
        while b <= NB2
            @inbounds begin
                shared_sums[b] = zero(FT)
                shared_cnts[b] = UInt32(0)
            end
            b += workgroup_size
        end
        @synchronize
    end
end

"""Kernel parameter list for `(dist_route, val_route)` joint tiled kernels."""
function _joint2d_kernel_param_exprs(dist_route::Symbol, val_route::Symbol)
    params = Any[:output_sums, :output_counts, :x_mat, :u_mat]
    if dist_route == :general
        push!(params, :( @Const(distance_edges) ))
    end
    if val_route == :general
        push!(params, :( @Const(value_edges) ))
    end
    append!(params, [
        :sf_type,
        :(N_points::Int),
        :(N_dist_edges::Int),
        :(N_val_edges::Int),
        :(NV::Int),
        :(NB2::Int),
    ])
    if dist_route == :linear
        append!(params, [
            :(first_edge::FT), :(last_edge::FT), :(inv_step::FT),
            :(step_val::FT),
        ])
    elseif dist_route == :log
        append!(params, [
            :(dist_first::FT), :(dist_last::FT), :(dist_inv_step::FT),
            :(dist_step::FT),
        ])
    elseif dist_route == :general
        push!(params, :(edge_anchor::FT))
    end
    if val_route == :linear || val_route == :log_linear
        append!(params, [
            :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
            :(val_step::FT),
        ])
    elseif val_route == :inflinear
        append!(params, [
            :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
            :(val_step::FT),
            :(n_inner_edges::Int), :(inner_last::FT),
        ])
    end
    append!(params, [:(sched), :(n_tile_blocks::Int), :(workgroup_size::Int), :(geom)])
    return params
end

function _joint2d_dist_digitize_expr(dist_route::Symbol)
    dist_route == :linear && return :(
        dbin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, step_val, N_dist_edges,
        )
    )
    dist_route == :log && return :(
        dbin = _gpu_digitize_log_spaced(
            dist, dist_first, dist_last, dist_inv_step, dist_step, N_dist_edges,
        )
    )
    return :(dbin = _gpu_digitize_general(dist, distance_edges, N_dist_edges))
end

function _joint2d_val_digitize_expr(val_route::Symbol)
    val_route == :general && return :(vbin = _gpu_digitize_general(val, value_edges, N_val_edges))
    val_route == :linear && return :(
        vbin = _gpu_digitize_linear(
            val, val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
    )
    val_route == :inflinear && return :(
        vbin = _gpu_digitize_inf_padded_linear(
            val, val_first, val_last, val_inv_step, val_step,
            n_inner_edges, inner_last,
        )
    )
    return :(
        vbin = _gpu_digitize_log_spaced(
            val, val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
    )
end

"""Codegen tiled joint kernel for `(dist_route, val_route, compile_cells)`."""
function _joint2d_kernel_def(dist_route::Symbol, val_route::Symbol, compile_cells::Int, W::Int)
    fname = _joint2d_kernel_fname(dist_route, val_route, compile_cells, W)
    zero_body = _joint2d_cooperative_zero_body()
    hist = compile_cells
    dist_digitize = _joint2d_dist_digitize_expr(dist_route)
    val_digitize = _joint2d_val_digitize_expr(val_route)
    params = _joint2d_kernel_param_exprs(dist_route, val_route)
    return quote
        KA.@kernel unsafe_indices=true function $(fname)($(params...),) where {FT}
            shared_xi = @localmem FT ($(W * SF_GPU_TILE),)
            shared_ui = @localmem FT ($(W * SF_GPU_TILE),)
            shared_xj = @localmem FT ($(W * SF_GPU_TILE),)
            shared_uj = @localmem FT ($(W * SF_GPU_TILE),)
            shared_sums = @localmem FT ($(hist),)
            shared_cnts = @localmem UInt32 ($(hist),)

            $(zero_body)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
            if bid <= n_tile_blocks
                ti, tj = tile_for(sched, bid)
                i0 = (ti - 1) * SF_GPU_TILE + 1
                j0 = (tj - 1) * SF_GPU_TILE + 1
                ni = min(SF_GPU_TILE, N_points - i0 + 1)
                nj = min(SF_GPU_TILE, N_points - j0 + 1)
                if ni > 0 && nj > 0
                    k = lid
                    while k <= ni
                        gi = i0 + k - 1
                        @inbounds begin
                            $(_joint2d_stage(:shared_xi, :x_mat, :gi, W))
                            $(_joint2d_stage(:shared_ui, :u_mat, :gi, W))
                        end
                        k += workgroup_size
                    end
                    if ti < tj
                        k = lid
                        while k <= nj
                            gj = j0 + k - 1
                            @inbounds begin
                                $(_joint2d_stage(:shared_xj, :x_mat, :gj, W))
                                $(_joint2d_stage(:shared_uj, :u_mat, :gj, W))
                            end
                            k += workgroup_size
                        end
                    end
                end
            end
            @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
            if bid <= n_tile_blocks
                ti, tj = tile_for(sched, bid)
                i0 = (ti - 1) * SF_GPU_TILE + 1
                j0 = (tj - 1) * SF_GPU_TILE + 1
                ni = min(SF_GPU_TILE, N_points - i0 + 1)
                nj = min(SF_GPU_TILE, N_points - j0 + 1)
                if ni > 0 && nj > 0
                    n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
                    p = lid
                    while p <= n_pairs
                        if ti < tj
                            ia = (p - 1) ÷ nj + 1
                            jb = (p - 1) - (ia - 1) * nj + 1
                            X1 = $(_joint2d_load(:shared_xi, :ia, W))
                            X2 = $(_joint2d_load(:shared_xj, :jb, W))
                            U1 = $(_joint2d_load(:shared_ui, :ia, W))
                            U2 = $(_joint2d_load(:shared_uj, :jb, W))
                        else
                            ia, jb = _pair_from_linear(p, ni)
                            X1 = $(_joint2d_load(:shared_xi, :ia, W))
                            X2 = $(_joint2d_load(:shared_xi, :jb, W))
                            U1 = $(_joint2d_load(:shared_ui, :ia, W))
                            U2 = $(_joint2d_load(:shared_ui, :jb, W))
                        end
                        ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                        $(dist_digitize)
                        if ok && 1 <= dbin < N_dist_edges
                            dU, r̂ = SFH.pair_increments(geom, frame, dist, X1, X2, U1, U2)
                            val = sf_type(dU, r̂)
                            $(val_digitize)
                            if 1 <= vbin < N_val_edges
                                idx = (dbin - 1) * NV + vbin
                                @atomic shared_sums[idx] += val
                                @atomic shared_cnts[idx] += UInt32(1)
                            end
                        end
                        p += workgroup_size
                    end
                end
            end
            @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
            if bid <= n_tile_blocks
                b = lid
                while b <= NB2
                    dbin = (b - 1) ÷ NV + 1
                    vbin = b - (dbin - 1) * NV
                    @atomic output_sums[dbin, vbin] += shared_sums[b]
                    if shared_cnts[b] != UInt32(0)
                        @atomic output_counts[dbin, vbin] += shared_cnts[b]
                    end
                    b += workgroup_size
                end
            end
        end
    end
end

"""Ensure a compiled tiled joint kernel exists; return callable."""
function _ensure_joint2d_kernel!(dist_route::Symbol, val_route::Symbol, compile_cells::Int, W::Int)
    key = (dist_route, val_route, compile_cells, W)
    if haskey(_JOINT2D_KERNEL_REGISTRY, key)
        return _JOINT2D_KERNEL_REGISTRY[key]
    end
    dist_route in _JOINT2D_DIST_ROUTES ||
        throw(ArgumentError("unknown joint2d dist route=$dist_route"))
    val_route in _JOINT2D_VAL_ROUTES ||
        throw(ArgumentError("unknown joint2d val route=$val_route"))
    compile_cells > 0 ||
        throw(ArgumentError("joint2d compile_cells must be positive (got $compile_cells)"))
    ex = _joint2d_kernel_def(dist_route, val_route, compile_cells, W)
    @eval $(ex)
    fname = _joint2d_kernel_fname(dist_route, val_route, compile_cells, W)
    _JOINT2D_KERNEL_REGISTRY[key] = Base.invokelatest(getfield, @__MODULE__, fname)
    return _JOINT2D_KERNEL_REGISTRY[key]
end

"""Resolve tiled joint kernel fn object for launch (compile on first use)."""
function _joint2d_tiled_kernel_fn(
    dist_route::Symbol,
    val_route::Symbol,
    compile_cells::Int,
    W::Int,
    backend::KA.Backend,
    ws::Int,
)
    kf = _ensure_joint2d_kernel!(dist_route, val_route, compile_cells, W)
    return Base.invokelatest(kf, backend, ws)
end

"""Launch a lazily compiled joint2d kernel (world-age safe)."""
function _joint2d_invoke_kernel!(kernel!, args...; ndrange)
    return Base.invokelatest(kernel!, args...; ndrange=ndrange)
end

# Both coordinate widths: 2 for flat 2-D and for the ambient sphere before conversion existed,
# 3 for the ambient sphere and for flat 3-D.
for dist_route in _JOINT2D_DIST_ROUTES, val_route in _JOINT2D_VAL_ROUTES, W in (2, 3)
    @eval $( _joint2d_kernel_def(dist_route, val_route, SF_GPU_MAX_2D_HIST, W) )
    _JOINT2D_KERNEL_REGISTRY[(dist_route, val_route, SF_GPU_MAX_2D_HIST, W)] =
        getfield(@__MODULE__, _joint2d_kernel_fname(dist_route, val_route, SF_GPU_MAX_2D_HIST, W))
end
