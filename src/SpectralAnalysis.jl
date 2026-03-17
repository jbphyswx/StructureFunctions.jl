"""
Module for spectral analysis of non-uniformly sampled data.
"""
module SpectralAnalysis

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Base.Threads: Threads

import ..HelperFunctions: HelperFunctions as HF

export calculate_spectrum, gpu_calculate_spectrum, DirectSumBackend, FINUFFTBackend, FFTBackend, plot_spectrum, compare_spectra, compare_spectral_analysis

abstract type AbstractSpectralBackend end
struct DirectSumBackend <: AbstractSpectralBackend end
struct FINUFFTBackend <: AbstractSpectralBackend end
struct FFTBackend <: AbstractSpectralBackend end

# Stub for GPU extension
function gpu_calculate_spectrum end

"""
    calculate_spectrum([backend], x_vecs::Tuple, u_vecs::Tuple, ms::Tuple; kwargs...)

Calculate the power spectrum for one or more fields.
"""
function calculate_spectrum(
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple;
    backend::AbstractSpectralBackend = DirectSumBackend(),
    kwargs...
)
    return calculate_spectrum(backend, x_vecs, u_vecs, ms; kwargs...)
end

function calculate_spectrum(
    backend::AbstractSpectralBackend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    eps::Real = 1e-9,
    domain_size::Union{Nothing, Tuple} = nothing
)
    # Validate and clean inputs
    D = length(x_vecs)
    NU = length(u_vecs)
    
    # Check consistency
    for d in 1:D
        @assert length(x_vecs[d]) == length(u_vecs[1]) "X and U length mismatch"
    end
    @assert length(ms) == D "Dimension of 'ms' must match 'x_vecs'"
    if domain_size !== nothing
        @assert length(domain_size) == D "Dimension of 'domain_size' must match 'x_vecs'"
    end

    return _calculate_spectrum(backend, x_vecs, u_vecs, ms, iflag, eps, domain_size)
end

# Internal dispatch for DirectSum
function _calculate_spectrum(
    ::DirectSumBackend,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    ms::NTuple{D, Int},
    iflag::Int,
    eps::Real,
    domain_size::Union{Nothing, Tuple} = nothing
) where {D, T1, T2}
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])
    NU = length(u_vecs)
    
    # 1. Coordinate ranges for physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(x_vecs[d])
            return max_x - min_x
        end
    end
    
    # Generate physical wavenumbers consistent with FINUFFT
    ks_phys = ntuple(d -> range(FT(-ms[d]÷2), stop=FT((ms[d]-1)÷2), length=ms[d]) .* (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])), Val(D))
    
    # Preallocate coefficients
    coeffs = zeros(Complex{FT}, ms..., NU)
    
    # O(N * M) 
    Threads.@threads for I in CartesianIndices(ntuple(i -> ms[i], Val(D)))
        # Current physical wavenumber vector
        k_phys = SA.SVector{D, FT}(ntuple(d -> ks_phys[d][I[d]], Val(D)))
        
        for j in 1:N
            x_pos = SA.SVector{D, FT}(ntuple(d -> x_vecs[d][j], Val(D)))
            
            # Phase factor (Direct Sum uses physical coordinates and physical wavenumbers)
            # Standard forward transform convention: e^(-i * k * x)
            phi = -iflag * (LA.dot(k_phys, x_pos))
            W = exp(LA.im * phi)
            
            for u_idx in 1:NU
                coeffs[I, u_idx] += u_vecs[u_idx][j] * W
            end
        end
    end
    
    # Scale by 1/N
    coeffs ./= N
    
    return coeffs, ks_phys
end


# Placeholder for extension methods
function _calculate_spectrum_nufft end

function _calculate_spectrum(
    ::FINUFFTBackend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
    domain_size::Union{Nothing, Tuple} = nothing
)
    return _calculate_spectrum_nufft(x_vecs, u_vecs, ms, iflag, eps, domain_size)
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
    domain_size::Union{Nothing, Tuple} = nothing
)
    return _calculate_spectrum_fft(x_vecs, u_vecs, ms, iflag, eps, domain_size)
end

"""
    plot_spectrum(k, c; kwargs...)

Plot the power spectrum magnitude. Requires `using CairoMakie`.
"""
function plot_spectrum end

"""
    compare_spectra(results; peaks=nothing, u_idx=1, kwargs...)

Compare multiple spectrum results in subplots. 
`results` is a list of "Label" => (k, c). 
`peaks` is an optional list of (kx, ky, amp) or (k, amp) to mark on the plots.
Requires `using CairoMakie`.
"""
function compare_spectra end

function compare_spectral_analysis end

end # module
