module StructureFunctionsFFTWExt

using FFTW: FFTW
using StructureFunctions: Calculations as SFC

"""An FFTW plan of host arrays runs on the threads the transform engine gives it."""
SFC._fft_plan_options(::Union{Array{<:FFTW.fftwNumber}, SubArray{<:FFTW.fftwNumber, <:Any, <:Array}}, threads::Int) =
    (; num_threads = threads)

end # module
