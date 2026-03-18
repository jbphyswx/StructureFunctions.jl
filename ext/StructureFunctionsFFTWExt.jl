module StructureFunctionsFFTWExt

using FFTW: FFTW
using StructureFunctions: StructureFunctions as SF, SpectralAnalysis as SFSA

"""
    _calculate_spectrum_fft(x_vecs, u_vecs, ms, iflag, eps; ...)

Implementation of spectral analysis using FFT (requires uniform grid).
"""
function SFSA._calculate_spectrum_fft(
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    eps::Real,
    domain_size::Union{Nothing, Tuple} = nothing,
)
    D = length(x_vecs)
    T1 = eltype(x_vecs[1])
    NU = length(u_vecs)
    FT = T1

    # FFTW expects a uniform grid. 
    # ms is the target grid size. 
    # For now we assume the data is already shaped as (N1, N2, ..., NU) 
    # but the API gives it to us as vectors. 
    # We reshaped it carefully in the test. 

    # 1. Reshape inputs to grid if they are vectors
    # (Assuming ms matches the underlying grid)
    coeffs = zeros(Complex{FT}, ms..., NU)
    for k in 1:NU
        uk = reshape(u_vecs[k], ms...)
        # FFTW.ifft is Type 2 (backwards), FFTW.fft is Type 1 (forwards)
        # Note: StructureFunctions standard iflag=1 is e^+ikx (backwards in FFTW)
        if iflag == 1
            coeffs_k = FFTW.fft(uk)
        else
            coeffs_k = FFTW.ifft(uk) .* prod(ms)
        end

        # 2. Shift to center zero-frequency
        # FFTW returns [0, M-1]. We need [-M/2, M/2-1].
        selectdim(coeffs, D + 1, k) .= FFTW.fftshift(coeffs_k)
    end

    # 3. Scaling
    # Average factor 1/N
    coeffs ./= prod(ms)

    # 4. Physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(x_vecs[d])
            return max_x - min_x
        end
    end
    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / ranges[d]),
        Val(D),
    )

    return coeffs, ks_phys
end

end # module
