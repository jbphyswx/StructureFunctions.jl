"""
Module for spectral analysis of non-uniformly sampled data.
"""
module SpectralAnalysis

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Base.Threads: Threads

import ..HelperFunctions: HelperFunctions as HF

export calculate_spectrum, DirectSumBackend, FINUFFTBackend, FFTBackend

abstract type AbstractSpectralBackend end
struct DirectSumBackend <: AbstractSpectralBackend end
struct FINUFFTBackend <: AbstractSpectralBackend end
struct FFTBackend <: AbstractSpectralBackend end

"""
    calculate_spectrum(x_vecs, u_vecs, ms; ...)

Calculate the Fourier coefficients of a non-uniformly sampled field `u_vecs` at coordinates `x_vecs`.
`ms` is a tuple of the number of modes in each dimension.

Returns:
- `coeffs`: Array of Fourier coefficients.
- `ks`: Tuple of wavenumber ranges.
"""
function calculate_spectrum(
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    ms::NTuple{D, Int};
    backend::AbstractSpectralBackend = DirectSumBackend(),
    iflag::Int = -1, # Standard FFT convention (e^-ikx)
    eps::Real = 1e-6,
) where {D, T1, T2}
    return _calculate_spectrum(backend, x_vecs, u_vecs, ms, iflag, eps)
end

# Internal dispatch for DirectSum
function _calculate_spectrum(
    ::DirectSumBackend,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    ms::NTuple{D, Int},
    iflag::Int,
    eps::Real,
) where {D, T1, T2}
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])
    
    # Identical to FINUFFT: Determine domain size for scaling
    ranges = ntuple(d -> extrema(x_vecs[d]), Val(D))
    L = ntuple(d -> ranges[d][2] - ranges[d][1], Val(D))
    
    # Generate physical wavenumbers consistent with FINUFFT
    ks = ntuple(i -> range(FT(-ms[i]÷2), stop=FT((ms[i]-1)÷2), length=ms[i]), Val(D))
    ks_phys = ntuple(i -> ks[i] .* (FT(2π) / (L[i] == 0 ? one(FT) : L[i])), Val(D))
    
    # Preallocate coefficients
    NU = length(u_vecs)
    coeffs = zeros(Complex{FT}, ms..., NU)
    
    # O(N * M) 
    Threads.@threads for I in CartesianIndices(ntuple(i -> ms[i], Val(D)))
        # Current physical wavenumber vector
        k_phys = SA.SVector{D, FT}(ntuple(d -> ks_phys[d][I[d]], Val(D)))
        
        for j in 1:N
            x_pos = SA.SVector{D, FT}(ntuple(d -> x_vecs[d][j], Val(D)))
            
            # Phase factor (Direct Sum uses physical coordinates and physical wavenumbers)
            phi = iflag * (LA.dot(k_phys, x_pos))
            W = exp(LA.im * phi)
            
            for u_idx in 1:NU
                coeffs[I, u_idx] += u_vecs[u_idx][j] * W
            end
        end
    end
    
    return coeffs, ks_phys
end


# Placeholder for extension
function _calculate_spectrum_nufft end

function _calculate_spectrum(
    ::FINUFFTBackend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
)
    return _calculate_spectrum_nufft(x_vecs, u_vecs, ms, iflag, eps)
end

# Placeholder for FFT extension
function _calculate_spectrum_fft end

function _calculate_spectrum(
    ::FFTBackend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
)
    return _calculate_spectrum_fft(x_vecs, u_vecs, ms, iflag, eps)
end

end
