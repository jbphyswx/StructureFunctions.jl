# Batch layout helpers — axis-agnostic trailing dimensions.

@inline function _require_trailing_batch(u::AbstractArray)
    ndims(u) >= 3 || throw(ArgumentError("expected ndims >= 3, got ndims=$(ndims(u))"))
    return u
end

"""Trailing index shape of `u` after leading `(N_dims, N)`."""
function batch_dims(u::AbstractArray)
    _require_trailing_batch(u)
    return size(u)[3:end]
end

"""Product of trailing batch dimensions."""
function batch_size(u::AbstractArray)
    return prod(batch_dims(u))
end

"""Linear batch index `b ∈ 1:batch_size` → tuple index into trailing dims."""
function batch_cartesian(b::Int, bdims::Dims)
    B = prod(bdims)
    (1 <= b <= B) || throw(ArgumentError("batch linear index $b not in 1:$B"))
    return Tuple(CartesianIndices(bdims)[b])
end

"""View `(N_dims, N)` field slice at linear batch index `b`."""
function batch_field_slice(u::AbstractArray, b::Int)
    bd = batch_dims(u)
    return @view u[:, :, batch_cartesian(b, bd)...]
end

"""Pad `x :: (N_dims, N)` and `u :: (N_dims, N, batch...)` to 3 × N (and 3 × N × batch...)."""
function pad3_batch(x_mat::AbstractMatrix{FT}, u_batch::AbstractArray{FT}) where {FT}
    _require_trailing_batch(u_batch)
    N_dims, N = size(x_mat)
    size(u_batch)[1:2] == (N_dims, N) ||
        throw(ArgumentError("u leading dims $(size(u_batch)[1:2]) must match x $(size(x_mat))"))
    bd = batch_dims(u_batch)
    x3 = zeros(FT, 3, N)
    x3[1:N_dims, :] .= x_mat
    u3 = zeros(FT, 3, N, bd...)
    B = prod(bd)
    u3_view = reshape(u3, 3, N, B)
    u_src = reshape(u_batch, N_dims, N, B)
    u3_view[1:N_dims, :, :] .= u_src
    return x3, u3, bd
end

"""Pad matching-rank `x`, `u` with trailing batch dims to 3 × N × batch..."""
function pad3_batch_matched(x_batch::AbstractArray{FT}, u_batch::AbstractArray{FT}) where {FT}
    _require_trailing_batch(x_batch)
    _require_trailing_batch(u_batch)
    size(x_batch) == size(u_batch) ||
        throw(ArgumentError("x and u must have the same shape for matched-rank batch"))
    N_dims, N = size(x_batch)[1:2]
    bd = batch_dims(x_batch)
    x3 = zeros(FT, 3, N, bd...)
    u3 = zeros(FT, 3, N, bd...)
    B = prod(bd)
    x3_view = reshape(x3, 3, N, B)
    u3_view = reshape(u3, 3, N, B)
    x_src = reshape(x_batch, N_dims, N, B)
    u_src = reshape(u_batch, N_dims, N, B)
    x3_view[1:N_dims, :, :] .= x_src
    u3_view[1:N_dims, :, :] .= u_src
    return x3, u3, bd
end

"""Allocate zeroed histogram buffers: `(NB, batch_dims...)` sums and counts."""
function allocate_batch_histogram(FT::Type, NB::Int, bdims::Dims)
    sums = zeros(FT, NB, bdims...)
    counts = zeros(UInt32, NB, bdims...)
    return sums, counts
end

function linear_bin_params(bin_edges::LinearBinEdges{FT}) where {FT}
    n_bins = length(bin_edges.edges)
    return (
        n_bins = n_bins,
        first_edge = bin_edges.first_edge,
        last_edge = bin_edges.last_edge,
        inv_step = bin_edges.inv_step,
        offset = bin_edges.offset,
        step_val = bin_edges.step_val,
    )
end

"""Which geometry case applies (rank/shape only)."""
@enum BatchGeometryCase begin
    FixedX
    VaryingX
end

function classify_batch_geometry(x::AbstractArray, u::AbstractArray)
    if ndims(x) == 2 && ndims(u) >= 3 && size(x, 2) == size(u, 2) && size(x, 1) == size(u, 1)
        return FixedX
    elseif ndims(x) >= 3 && size(x) == size(u)
        return VaryingX
    else
        throw(ArgumentError(
            "unsupported batch geometry: ndims(x)=$(ndims(x)), ndims(u)=$(ndims(u)), " *
            "size(x)=$(size(x)), size(u)=$(size(u)); " *
            "expected x::(N_dims,N) with u trailing batch, or x and u same shape",
        ))
    end
end

@inline function _read_u3(u3::AbstractArray{FT}, i::Int, b::Int, bdims::Dims) where {FT}
    ci = batch_cartesian(b, bdims)
    return SA.SVector{3, FT}(u3[1, i, ci...], u3[2, i, ci...], u3[3, i, ci...])
end

@inline function _read_x3(x3::AbstractArray{FT}, i::Int, b::Int, bdims::Dims) where {FT}
    ci = batch_cartesian(b, bdims)
    return SA.SVector{3, FT}(x3[1, i, ci...], x3[2, i, ci...], x3[3, i, ci...])
end

@inline function _pair_geometry(x3::AbstractMatrix{FT}, i::Int, j::Int) where {FT}
    X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
    X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    r̂ = SFH.r̂(X1, X2)
    return dist, r̂
end

@inline function _pair_geometry(x3::AbstractArray{FT}, i::Int, j::Int, b::Int, bdims::Dims) where {FT}
    X1 = _read_x3(x3, i, b, bdims)
    X2 = _read_x3(x3, j, b, bdims)
    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    r̂ = SFH.r̂(X1, X2)
    return dist, r̂
end

@inline function _digitize_dist(dist, lp)
    return _GPUP._gpu_digitize_linear(
        dist, lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val, lp.n_bins,
    )
end

@inline function _in_histogram_bin(bin::Int, lp)
    return 1 <= bin < lp.n_bins
end

"""Read velocity `(u,v)` at batch index `b` and grid point `gi` from device `(B, N, 2)` layout."""
@inline function _batch_u_at(u_batch::AbstractArray{FT}, b::Int, gi::Int) where {FT}
    return SA.SVector{2, FT}(u_batch[b, gi, 1], u_batch[b, gi, 2])
end

@inline function _accumulate_bin!(sums, counts, bin::Int, b::Int, val, bdims::Dims)
    ci = batch_cartesian(b, bdims)
    sums[bin, ci...] += val
    counts[bin, ci...] += one(UInt32)
    return nothing
end

function _flatten_sums_counts(sums, counts)
    NB = size(sums, 1)
    return reshape(sums, NB, :), reshape(counts, NB, :)
end

function max_abs_diff(a, b)
    va = vec(a)
    vb = vec(b)
    return isempty(va) ? 0.0 : maximum(abs.(va .- vb))
end

function histograms_equal(sums_a, counts_a, sums_b, counts_b; rtol = 1e-4, atol = 1f-3)
    ref = maximum(abs.(vec(sums_b)); init = 0f0)
    sums_ok = max_abs_diff(sums_a, sums_b) <= atol + rtol * ref
    counts_ok = counts_a == counts_b
    return sums_ok && counts_ok
end

"""Print profile line to stdout; append to `test/debug/batch_profile.log` when `BATCH_PROFILE=1`."""
function _batch_profile_log!(msg::AbstractString)
    println(msg)
    flush(stdout)
    get(ENV, "BATCH_PROFILE", "") == "1" || return
    log_dir = abspath(joinpath(@__DIR__, "..", "..", "test", "debug"))
    mkpath(log_dir)
    open(joinpath(log_dir, "batch_profile.log"), "a") do io
        println(io, msg)
    end
    return nothing
end
