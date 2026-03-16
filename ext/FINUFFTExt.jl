module FINUFFTExt

import FINUFFT
import StructureFunctions.SpectralAnalysis as SA_mod
import StructureFunctions.SpectralAnalysis: FINUFFTBackend

"""
    _calculate_spectrum_nufft(x_vecs, u_vecs, ms, iflag, eps; ...)

Implementation of spectral analysis using Non-Uniform Fast Fourier Transform (FINUFFT).
"""
function SA_mod._calculate_spectrum_nufft(
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
    domain_size::Union{Nothing, Tuple} = nothing
)
    D = length(x_vecs)
    NU = length(u_vecs)
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])

    # 1. Filter out NaNs/Infs
    valid_mask = trues(N)
    for d in 1:D
        valid_mask .&= isfinite.(x_vecs[d])
    end
    for k in 1:NU
        valid_mask .&= isfinite.(u_vecs[k])
    end
    
    N_valid = count(valid_mask)
    if N_valid == 0
        return zeros(Complex{FT}, ms..., NU), ntuple(i -> range(zero(FT), stop=zero(FT), length=ms[i]), Val(D))
    end

    scaled_x = ntuple(Val(D)) do d
        x = x_vecs[d][valid_mask]
        min_x, max_x = extrema(x)
        range_calc = max_x - min_x
        range_x = domain_size === nothing ? range_calc : domain_size[d]
        
        if range_x ≈ 0
            return (zeros(FT, N_valid), min_x, one(FT))
        end
        # Map to [0, 2π] using the effective domain size
        return (FT(2π) .* (x .- min_x) ./ range_x, min_x, range_x)
    end
    
    xs = ntuple(i -> scaled_x[i][1], Val(D))
    offsets = ntuple(i -> scaled_x[i][2], Val(D))
    ranges_calc = ntuple(i -> scaled_x[i][3], Val(D))
    ranges = domain_size === nothing ? ranges_calc : domain_size

    # 3. Call FINUFFT
    # Result is ms... x NU
    coeffs = zeros(Complex{FT}, ms..., NU)
    
    # FINUFFT Type 1: Non-uniform to uniform
    for k in 1:NU
        uk = Complex{FT}.(u_vecs[k][valid_mask])
        # FINUFFT.jl is picky about contiguous Arrays for output
        coeffs_k = zeros(Complex{FT}, ms...)
        
        if D == 1
            FINUFFT.nufft1d1!(xs[1], uk, iflag, eps, coeffs_k)
        elseif D == 2
            FINUFFT.nufft2d1!(xs[1], xs[2], uk, iflag, eps, coeffs_k)
        elseif D == 3
            FINUFFT.nufft3d1!(xs[1], xs[2], xs[3], uk, iflag, eps, coeffs_k)
        else
            error("FINUFFT only supports up to 3D")
        end
        Base.selectdim(coeffs, D+1, k) .= coeffs_k
    end

    # 4. Phase shift to account for min_x offset (Translation Property)
    # If we mapped x -> (x - min_x) * 2π/L, we must multiply by exp(ik * min_x * 2π/L)
    ks = ntuple(i -> range(FT(-ms[i]÷2), stop=FT((ms[i]-1)÷2), length=ms[i]), Val(D))
    
    for d in 1:D
        k_vec = ks[d]
        L = ranges[d]
        phase = exp.(im .* iflag .* k_vec .* (offsets[d] * FT(2π) / L))
        if D == 1
            for k in 1:NU
                coeffs[:, k] .*= phase
            end
        elseif D == 2
            if d == 1
                for k in 1:NU, j in 1:ms[2]
                    coeffs[:, j, k] .*= phase
                end
            else
                for k in 1:NU, i in 1:ms[1]
                    coeffs[i, :, k] .*= phase
                end
            end
        end
    end

    # 5. Scaling
    coeffs ./= N_valid

    # 6. Physical wavenumbers
    ks_phys = ntuple(i -> ks[i] .* (FT(2π) / (ranges[i] == 0 ? one(FT) : ranges[i])), Val(D))

    return coeffs, ks_phys
end

end # module
