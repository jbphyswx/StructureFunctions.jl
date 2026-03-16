module FFTWExt

using FFTW: FFTW
using StaticArrays: StaticArrays as SA
import StructureFunctions.SpectralAnalysis: SpectralAnalysis as SA_mod

"""
    _calculate_spectrum_fft(x_vecs, u_vecs, ms, iflag, eps)

Extension implementation for FFTW backend.
Calculates FFT.
"""
function SA_mod._calculate_spectrum_fft(
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
)
    D = length(x_vecs)
    T1 = eltype(x_vecs[1])
    T2 = eltype(u_vecs[1])
    FT = T1
    N = length(x_vecs[1])
    NU = length(u_vecs)

    # Preallocate uniform data
    coeffs = zeros(Complex{FT}, ms..., NU)
    
    for u_idx in 1:NU
        # If D=1:
        if D == 1
            # Check if input length matches 'ms' (uniform grid assumption)
            if N == ms[1]
                data = u_vecs[u_idx]
                if iflag == 1
                    selectdim(coeffs, D + 1, u_idx) .= FFTW.fft(data)
                else
                    # Scale to match sum-convention
                    selectdim(coeffs, D + 1, u_idx) .= FFTW.ifft(data) .* N
                end
            else
                error("FFTBackend currently requires input length to match 'ms' (uniform grid assumption)")
            end
        else
            error("FFTBackend only supports 1D currently. Use FINUFFT or DirectSum for nD scattered data.")
        end
    end

    ranges = ntuple(d -> extrema(x_vecs[d]), Val(D))
    ks_phys = ntuple(d -> range(FT(-ms[d]÷2), stop=FT((ms[d]-1)÷2), length=ms[d]) .* (FT(2π) / (ranges[d][2] - ranges[d][1])), Val(D))

    return coeffs, ks_phys
end

end
