module FINUFFTExt

using FINUFFT: FINUFFT
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
import StructureFunctions.SpectralAnalysis: SpectralAnalysis as SA_mod

"""
    _calculate_spectrum_nufft(backend, x_vecs, u_vecs, ms, iflag, eps)

Extension implementation for FINUFFT backend.
"""
function SA_mod._calculate_spectrum_nufft(
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    ms::NTuple{D, Int},
    iflag::Int,
    eps::Real,
) where {D, T1, T2}
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])
    NU = length(u_vecs)

    # 1. Handle NaNs: Drop points that are NaN in any coordinate or any component
    # For now, let's keep it simple and drop points that are NaN in x or any of u.
    valid_mask = trues(N)
    for d in 1:D
        valid_mask .&= .!isnan.(x_vecs[d])
    end
    for u_idx in 1:NU
        valid_mask .&= .!isnan.(u_vecs[u_idx])
    end
    
    N_valid = sum(valid_mask)
    if N_valid == 0
        return zeros(Complex{FT}, ms..., NU), ntuple(i -> range(zero(FT), stop=zero(FT), length=ms[i]), Val(D))
    end

    # 2. Extract and Scale Coordinates
    # FINUFFT expects points in [0, 2π] (for type 1)
    scaled_x = ntuple(Val(D)) do d
        x = x_vecs[d][valid_mask]
        min_x, max_x = extrema(x)
        range_x = max_x - min_x
        if range_x ≈ 0
            return (zeros(FT, N_valid), min_x, one(FT))
        end
        # Map to [0, 2π]
        return (FT(2π) .* (x .- min_x) ./ range_x, min_x, range_x)
    end
    
    xs = ntuple(i -> scaled_x[i][1], Val(D))
    offsets = ntuple(i -> scaled_x[i][2], Val(D))
    ranges = ntuple(i -> scaled_x[i][3], Val(D))

    # 3. Call FINUFFT
    # Result is ms... x NU
    coeffs = zeros(Complex{FT}, ms..., NU)
    
    # FINUFFT interface varies by dimension
    # We'll use the most generic one available.
    # For now, let's implement 1D, 2D, 3D specifically as they are most common.
    for u_idx in 1:NU
        u_data = Complex{FT}.(u_vecs[u_idx][valid_mask])
        if D == 1
            coeffs_u = FINUFFT.nufft1d1(xs[1], u_data, iflag, eps, ms[1])
            selectdim(coeffs, D + 1, u_idx) .= coeffs_u
        elseif D == 2
            coeffs_u = FINUFFT.nufft2d1(xs[1], xs[2], u_data, iflag, eps, ms[1], ms[2])
            selectdim(coeffs, D + 1, u_idx) .= coeffs_u
        elseif D == 3
            coeffs_u = FINUFFT.nufft3d1(xs[1], xs[2], xs[3], u_data, iflag, eps, ms[1], ms[2], ms[3])
            selectdim(coeffs, D + 1, u_idx) .= coeffs_u
        else
            error("FINUFFT extension only supports 1D, 2D, and 3D currently.")
        end
    end

    # 4. Correct for Scaling and Translation
    # FINUFFT calculates: Σ u_j exp(im * iflag * k * x_scaled_j)
    # where x_scaled = 2π/L * (x - min_x)
    # So phase = im * iflag * k * (2π/L) * (x - min_x)
    # We want phase = im * iflag * k_phys * x
    # To match, k_phys = k * (2π/L)
    # And we have an extra factor exp(-im * iflag * k * (2π/L) * min_x) that we need to cancel?
    # No, the result is Σ u_j exp(im * iflag * k * x_scaled_j).
    # We want to return physical coefficients and wavenumbers.
    
    # Physical wavenumbers ks_phys = k * (2π / ranges)
    ks_phys = ntuple(d -> range(FT(-ms[d]÷2), stop=FT((ms[d]-1)÷2), length=ms[d]) .* (FT(2π) / ranges[d]), Val(D))
    
    # Phase shift to account for the -min_x translation
    # FT{f(x - a)} = exp(-im * k_phys * a) * FT{f(x)}
    # Here our "original" signal is f(x). We are calculating FT of f(x_scaled).
    # Actually, it's easier to think of it as:
    # exp(im * iflag * k * x_scaled) = exp(im * iflag * k * (2π/L) * (x - min_x))
    # = exp(im * iflag * k_phys * x) * exp(-im * iflag * k_phys * min_x)
    # So to get the true FT (Σ u_j exp(im * iflag * k_phys * x_j)), we multiply by exp(im * iflag * k_phys * min_x).
    
    for I in CartesianIndices(ntuple(i -> ms[i], Val(D)))
        k_phys_vec = SA.SVector{D, FT}(ntuple(d -> ks_phys[d][I[d]], Val(D)))
        total_offset_phase = zero(FT)
        for d in 1:D
            total_offset_phase += k_phys_vec[d] * offsets[d]
        end
        # Correction factor
        shift = exp(im * iflag * total_offset_phase)
        for u_idx in 1:NU
            coeffs[I, u_idx] *= shift
        end
    end

    return coeffs, ks_phys
end

end
