# The sorted line on a device: the points sorted once by coordinate, the centred field's weighted monomials
# prefix-summed in the line's prefix type, and one work item per point summing each bin's partner range.

"""Lanes of a sorted-line workgroup."""
const LINE_WG = 128

KA.@kernel function _line_monomials!(mono, @Const(ds), centre, weights, ::Val{W}, ::Val{P}) where {W, P}
    j = @index(Global, Linear)
    T = eltype(mono)
    u = SA.SVector{W, T}(ntuple(c -> T(@inbounds(ds[c, j])) - centre[c], Val(W)))
    m = SFC._line_monomials(u, Val(W), Val(P)) * T(SFC._point_weight(weights, j))
    for k in eachindex(m)
        @inbounds mono[k, j] = m[k]
    end
end

KA.@kernel unsafe_indices = true function _line_sweep!(
    sums, counts, sf, @Const(xs), @Const(ds), @Const(S), centre, weights, plan, nb::Int, N::Int, r̂,
    ::Val{W}, ::Val{P}, ::Val{V}, ::Val{K}, ::Val{NK}, ::Val{SHARED},
) where {W, P, V, K, NK, SHARED}
    acc_s = @localmem eltype(sums) (SF_GPU_MAX_BINS,)
    acc_c = @localmem eltype(counts) (SF_GPU_MAX_BINS,)
    lid = @index(Local, Linear)
    if SHARED
        k = Int(lid)
        while k <= nb
            @inbounds acc_s[k] = zero(eltype(sums))
            @inbounds acc_c[k] = zero(eltype(counts))
            k += LINE_WG
        end
    end
    @synchronize

    gi = @index(Global, Linear)
    i = Int(gi)
    if i < N
        T = eltype(S)
        ui = SA.SVector{W, T}(ntuple(c -> T(@inbounds(ds[c, i])) - centre[c], Val(W)))
        μi = SFC._line_monomials(ui, Val(W), Val(P))
        wi = T(SFC._point_weight(weights, i))
        q = SFC._line_partner_bound(xs, plan, i, 0, N)
        for b in 1:nb
            qb = SFC._line_partner_bound(xs, plan, i, b, N, q)
            lo = max(q, i)
            if qb > lo
                v = eltype(sums)(SFC._line_range_value(sf, S, μi, wi, lo, qb, r̂, Val(W), Val(P), Val(V), Val(K),
                                                       Val(NK)))
                n = SFC._range_count(eltype(counts), weights, wi, S, lo, qb)
                if SHARED
                    @atomic acc_s[b] += v
                    @atomic acc_c[b] += n
                else
                    @atomic sums[b] += v
                    @atomic counts[b] += n
                end
            end
            q = qb
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    if SHARED
        k = Int(lid)
        while k <= nb
            cv = @inbounds acc_c[k]
            if !iszero(cv)
                @atomic sums[k] += @inbounds acc_s[k]
                @atomic counts[k] += cv
            end
            k += LINE_WG
        end
    end
end

"""The weighted mean of each row of the device field `ds` as an `SVector{W, PT}`, column `j` weighted by `w[j]`."""
function _line_device_centre(ds, ::SFC.NoWeights, ::Type{PT}, ::Val{W}) where {PT, W}
    h = Array(sum(x -> PT(x), ds; dims = 2))
    return SA.SVector{W, PT}(ntuple(c -> h[c] / size(ds, 2), Val(W)))
end

function _line_device_centre(ds, w::AbstractVector, ::Type{PT}, ::Val{W}) where {PT, W}
    h = Array(sum(PT.(ds) .* reshape(PT.(w), 1, :); dims = 2))
    total = PT(sum(w))
    return SA.SVector{W, PT}(ntuple(c -> h[c] / total, Val(W)))
end

"""
    _gpu_sorted_line!(sums, counts, backend, sf, x, data, distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, weights)

Add the pair statistic of a polynomial operator over points on a line into the device `sums`/`counts`, as
[`SFC.sorted_line_sweep!`](@ref) does on the host: `x` one coordinate per point, `data` the packed fields
`(V·D + K, N)`, `weights` `NoWeights()` or one per point, on the host or on `backend`.
"""
function _gpu_sorted_line!(sums, counts, backend::KA.Backend, sf, x::AbstractVector, data::AbstractMatrix,
                           distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, weights) where {D, V, K}
    N = length(x)
    N < 2 && return nothing
    W, P = V * D + K, SFT.order(sf)
    PT = SFC._line_prefix_type(eltype(sums))
    to = a -> KA.adapt(backend, a)
    xd = to(x)
    perm = _gpu_sortperm(xd)
    xs, ds = @inbounds(xd[perm]), @inbounds(to(data)[:, perm])
    ws = weights isa SFC.NoWeights ? weights : @inbounds(to(weights)[perm])
    centre = _line_device_centre(ds, ws, PT, Val(W))
    vNK = SFC._monomial_count(Val(W), Val(P))
    mono = KA.allocate(backend, PT, SFC._val_int(vNK), N)
    _line_monomials!(backend, LINE_WG)(mono, ds, centre, ws, Val(W), Val(P); ndrange = N)
    S = KA.zeros(backend, PT, SFC._val_int(vNK), N + 1)
    cumsum!(view(S, :, 2:(N + 1)), mono; dims = 2)
    nb = SFC.n_histogram_bins(distance_bins)
    r̂ = SFC._unit_line(SFC._direction_width(Val(D), Val(V), Val(1)), PT)
    _line_sweep!(backend, LINE_WG)(sums, counts, sf, xs, ds, S, centre, ws,
                                   _dist_digitizer(nothing, backend, distance_bins, Val(:sf1d)), nb, N, r̂,
                                   Val(W), Val(P), Val(V), Val(K), vNK, Val(nb <= SF_GPU_MAX_BINS);
                                   ndrange = cld(N - 1, LINE_WG) * LINE_WG)
    return nothing
end
