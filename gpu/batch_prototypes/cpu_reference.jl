# CPU reference paths: slice baseline and one-pass batched prototypes.

"""
    cpu_slice_baseline!(sums, counts, x, u, sf_type, bin_edges; fixed_x=true)

Loop over linear batch indices; run single-snapshot gold histogram per slice.
`fixed_x=true`: `x` is `(N_dims, N)`; `fixed_x=false`: `x` matches `u` shape.
"""
function cpu_slice_baseline!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    fixed_x::Bool = true,
) where {FT}
    bd = batch_dims(u)
    B = batch_size(u)
    sums_f, counts_f = _flatten_sums_counts(sums, counts)
    for b in 1:B
        if fixed_x
            x_slice = x
            u_slice = batch_field_slice(u, b)
        else
            x_slice = batch_field_slice(x, b)
            u_slice = batch_field_slice(u, b)
        end
        gold = _GPUP.cpu_gold_histogram(x_slice, u_slice, sf_type, bin_edges)
        sums_f[:, b] .= gold.sums
        counts_f[:, b] .= UInt32.(gold.counts_i64)
    end
    return nothing
end

"""
    cpu_batch_fixed_x!(sums, counts, x_mat, u_batch, sf_type, bin_edges; strip_width=32)

Case A: fixed geometry matrix `x`, batched `u`. One pair traversal; vectorized batch strips.
"""
function cpu_batch_fixed_x!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    strip_width::Int = 32,
) where {FT}
    lp = linear_bin_params(bin_edges)
    x3, u3, bd = pad3_batch(x_mat, u_batch)
    N = size(x_mat, 2)
    B = batch_size(u_batch)
    _GPUP.verify_pair_enumeration(N)
    total_pairs = N * (N - 1) ÷ 2
    fill!(sums, zero(FT))
    fill!(counts, zero(UInt32))

    @inbounds for k in 1:total_pairs
        i, j = _GPUP._pair_from_linear(k, N)
        dist, r̂ = _pair_geometry(x3, i, j)
        bin = _digitize_dist(dist, lp)
        _in_histogram_bin(bin, lp) || continue
        b0 = 1
        while b0 <= B
            b1 = min(b0 + strip_width - 1, B)
            @inbounds @simd for b in b0:b1
                U1 = _read_u3(u3, i, b, bd)
                U2 = _read_u3(u3, j, b, bd)
                val = sf_type(U2 - U1, r̂)
                _accumulate_bin!(sums, counts, bin, b, val, bd)
            end
            b0 = b1 + 1
        end
    end
    return nothing
end

"""
    cpu_batch_varying_x!(sums, counts, x_batch, u_batch, sf_type, bin_edges; strip_width=32)

Case B: `x` and `u` same shape with trailing batch dims. Geometry recomputed per batch element.
"""
function cpu_batch_varying_x!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    strip_width::Int = 32,
) where {FT}
    lp = linear_bin_params(bin_edges)
    x3, u3, bd = pad3_batch_matched(x_batch, u_batch)
    N = size(x_batch, 2)
    B = batch_size(u_batch)
    _GPUP.verify_pair_enumeration(N)
    total_pairs = N * (N - 1) ÷ 2
    fill!(sums, zero(FT))
    fill!(counts, zero(UInt32))

    @inbounds for k in 1:total_pairs
        i, j = _GPUP._pair_from_linear(k, N)
        b0 = 1
        while b0 <= B
            b1 = min(b0 + strip_width - 1, B)
            @inbounds @simd for b in b0:b1
                dist, r̂ = _pair_geometry(x3, i, j, b, bd)
                bin = _digitize_dist(dist, lp)
                if _in_histogram_bin(bin, lp)
                    U1 = _read_u3(u3, i, b, bd)
                    U2 = _read_u3(u3, j, b, bd)
                    val = sf_type(U2 - U1, r̂)
                    _accumulate_bin!(sums, counts, bin, b, val, bd)
                end
            end
            b0 = b1 + 1
        end
    end
    return nothing
end
