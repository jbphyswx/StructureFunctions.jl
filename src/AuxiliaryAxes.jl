# Trailing auxiliary-axis helper functions (shapes/indexing).

"""Trailing auxiliary-axis shape of `u` after leading `(D, N)`."""
function batch_dims(u::AbstractArray)
    ndims(u) >= 3 || throw(ArgumentError("expected ndims(u) >= 3, got ndims=$(ndims(u))"))
    return size(u)[3:end]
end

"""Product of trailing auxiliary-axis dimensions."""
function batch_size(u::AbstractArray)
    return prod(batch_dims(u))
end

"""View `(D, N)` field slice at linear auxiliary index `b`."""
function batch_field_slice(u::AbstractArray, b::Int)
    bd = batch_dims(u)
    # Reshape and take view
    N_dims, N = size(u)[1:2]
    B = prod(bd)
    u_flat = reshape(u, N_dims, N, B)
    return @view u_flat[:, :, b]
end

function _flatten_sums_counts(sums, counts)
    NB = size(sums, 1)
    return reshape(sums, NB, :), reshape(counts, NB, :)
end

function batch_max_abs_diff(a, b)
    va = vec(a)
    vb = vec(b)
    return isempty(va) ? 0.0 : maximum(abs.(va .- vb))
end

function batch_histograms_equal(sums_a, counts_a, sums_b, counts_b; rtol = 1e-4, atol = 1f-3)
    ref = maximum(abs.(vec(sums_b)); init = 0f0)
    sums_ok = batch_max_abs_diff(sums_a, sums_b) <= atol + rtol * ref
    counts_ok = counts_a == counts_b
    return sums_ok && counts_ok
end

"""Map 1-based upper-triangle pair index to `(i, j)` with `i < j` for `N` points."""
@inline function _pair_from_linear(k::Int, N::Int)
    (1 <= k <= N * (N - 1) ÷ 2) ||
        throw(ArgumentError("pair index k=$k out of range for N=$N"))
    k64 = Int64(k)
    N64 = Int64(N)
    term = 4 * N64 * N64 - 4 * N64 + 1 - 8 * (k64 - 1)
    i = floor(Int, (2 * N64 - 1 - sqrt(Float64(term))) / 2) + 1
    pairs_before = (i - 1) * N - (i - 1) * i ÷ 2
    j = i + (k - pairs_before)
    return i, j
end

"""Public alias for tests and callers enumerating upper-triangle pairs."""
pair_from_linear(k::Int, N::Int) = _pair_from_linear(k, N)
