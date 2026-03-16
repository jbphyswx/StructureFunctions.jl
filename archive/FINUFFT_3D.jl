# I wanna test if I can get FINUFFT to take a 2D NUFFT

using FINUFFT # see https://github.com/ludvigak/FINUFFT.jl, https://ludvigak.github.io/FINUFFT.jl/latest, https://finufft.readthedocs.io/en/latest/math.html 
using StaticArrays

# 2D NUFFT type 1

function test_2d_nufft_3D(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{NTuple{N2, FT2}, SVector{N2, FT2}, AbstractVector{FT2}}}; # Tuple{Vararg{Vector{FT},N}}
    ms::Int = 128,
    mt::Int = 128,
) where {N, N2, FT1, FT2}
    """
    Test 2D NUFFT
    - using type1 so we get unifrom outputs (i think that's faster and/or more accurate?)
    """
   error("Not implemented yet")
end


#### should we really be taking FFT(KE) rather than of u,v?
## Then we dont have bins to worry about and we can just do ∫E(k)dk (should be 2d? so just πk^2) → E(k) * k^2 at each k

function test_2d_nufft_3D(
    x_vecs::AbstractVector{ <:Union{NTuple{N2, FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # testing if vector avoids repeated compilation.... doesn't seem to lol... idk
    # u_vecs::AbstractVector{ <:Union{ SVector{NT, NTuple{N2, FT2}}, NTuple{NT, NTuple{N2, FT2}}, AbstractVector{NTuple{N2, FT2}},  AbstractVector{AbstractVector{FT2}} }};
    u_vecs::AbstractArray{FT2}; # 3D AbstractArray
    ms::Int64 = 128,
    mt::Int64 = 128,
    iflag::Int64 = 1,
    eps::FT1 = FT1(1e-6),
) where {N2 <:Int64, FT1 <: AbstractFloat, FT2 <: AbstractFloat, NT}
    """
    Test 2D NUFFT
    - using type1 so we get unifrom outputs (i think that's faster and/or more accurate?)





    NOTE!!!! these all have  the same k rn for the times given to this func but each calculation receives a different set of points in x_vec since we filter in advance lol... 
    We NEED to set up the version where you can pass in the k you want ahead of time!!!
    """

    if !@isdefined(NT)
        NTT::Int64 = size(u_vecs, 2)
    else
        NTT = NT
    end

    # we cannot combine the third dimension like w/ SFs since we have points with the same coordinates...
    # To Do - preallocate output location...

    x, y = eachrow( reduce( hcat, (map(Array, x_vecs)))); # this line is slow bc of precompilation, precompiles for every different length of x_vecs because why?
    x = Array(x); # view to array
    y = Array(y); # view to array
    # functions to scale data to lie in [-3π, 3π] × [-3π, 3π] and revert that scaling back to original 
    min_x, max_x = extrema(x)
    min_y, max_y = extrema(y)
    # scaling(x) = eltype(x)(2π+0) * (x .- minimum(x)) / (maximum(x) - minimum(x)) # we use [0, 2π] as our range
    x = eltype(x)(2π+0) * (x .- min_x) / (max_x - min_x)
    y = eltype(y)(2π+0) * (y .- min_y) / (max_y - min_y)

    k_x = range(FT1(-ms/2), stop=FT1((ms-1)/2), length=ms) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    k_y = range(FT1(-mt/2), stop=FT1((mt-1)/2), length=mt) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1

    # scale our wavenumbers to get back the original wavenumbers ( from 0 => 2π to 0 => max_x-min_x ) basically going e^ikx to e^i(k/a)x = e^ik(x/a) fpr some constant a
    k_x = k_x .* FT1(2π) / (max_x - min_x)
    k_y = k_y .* FT1(2π) / (max_y - min_y)

    # after scaling k, we have data representing x from  0 to max_x-min_x and need to shift to x from min_x to max_x.., so we shift right by min_x, and e^ik(x+shift) = e^ikx * e^ik(shift),  (this doesnt change the amplitude and thus the energy spectrum...)
    # shift_x = exp.(-1im * k_x * min_x) # the fft of the shift in real space... 
    # shift_y = exp.(-1im * k_y * min_y)
    shift = exp.(-1im * k_x * min_x) .* exp.(-1im * k_y * min_y)' # the fft of the shift in real space... (one of these should be transposed?)

    nufft_us = zeros(Complex{FT2}, ms, mt ,1 , NTT)
    nufft_vs = zeros(Complex{FT2}, ms, mt ,1 , NTT)

    # valid::MVector{length(x), Bool} = trues(size(x))# could use a MVector (incredibly slow for precompilatino which is bad bc length(x_vecs) changes all the time if we filter out nans in python...
    valid::BitVector = trues(size(x))# could use a MVector (incredibly slow for precompilatino which is bad bc length(x_vecs) changes all the time if we filter out nans in python...

    for k in Base.OneTo(NTT::Int64)
        u = view(u_vecs, :, k, 1)
        v = view(u_vecs, :, k, 2)
        # drop nans (bc we can only drop points that are always nan, but many points are nan sometimes
        valid .= .!isnan.(u) .& .!isnan.(v)

        # short circuit if all are nan
        if sum(valid) == 0 # sum is allegedly the fastest
            nufft_us[:, :, 1, k] .= Complex{FT2}(NaN)
            nufft_vs[:, :, 1, k] .= Complex{FT2}(NaN)
        else

            u = Array{Complex{eltype(u)}}(u); # view to array
            v = Array{Complex{eltype(v)}}(v); # view to array
        
            # NUFFT object
            nufft_u = FINUFFT.nufft2d1(x[valid], y[valid], u[valid], iflag, eps, ms, ms) # would use views but FINUFFT doesn't like it
            nufft_v = FINUFFT.nufft2d1(x[valid], y[valid], v[valid], iflag, eps, ms, mt)

            # apply the fft'd shift to our wavenumber coefficients bc e^ikx should be shifted by the algorithm limited us to 0 to 2pi
            nufft_u .*= shift  # this is an ms x mt x 1 array
            nufft_v .*= shift

            # Return result
            nufft_us[:, :, 1, k] .= nufft_u
            nufft_vs[:, :, 1, k] .= nufft_v
        end

        
    end
    return nufft_us::Array{Complex{FT2},4}, nufft_vs::Array{Complex{FT2},4}, k_x, k_y # output was type unstable I guess?
    # return nufft_us::Array{Complex{FT2},3}, nufft_v::Array{Complex{FT2},3}, k_x, k_y # output was type unstable I guess?
end





function test_2d_nufft_3D_KE(
    x_vecs::AbstractVector{ <:Union{NTuple{N2, FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # testing if vector avoids repeated compilation.... doesn't seem to lol... idk
    KE_vec::AbstractArray{FT2}; # 3D AbstractArray
    ms::Int64 = 128,
    mt::Int64 = 128,
    iflag::Int64 = 1,
    eps::FT1 = FT1(1e-6),
) where {N2 <:Int64, FT1 <: AbstractFloat, FT2 <: AbstractFloat, NT}
    """
    Test 2D NUFFT
    - using type1 so we get unifrom outputs (i think that's faster and/or more accurate?)


    NOTE!!!! these all have  the same k rn for the times given to this func but each calculation receives a different set of points in x_vec since we filter in advance lol... 
    We NEED to set up the version where you can pass in the k you want ahead of time!!!
    """

    if !@isdefined(NT)
        NTT::Int64 = size(KE_vec, 2)
    else
        NTT = NT
    end

    # we cannot combine the third dimension like w/ SFs since we have points with the same coordinates...
    # To Do - preallocate output location...

    x, y = eachrow( reduce( hcat, (map(Array, x_vecs)))); # this line is slow bc of precompilation, precompiles for every different length of x_vecs because why?
    x = Array(x); # view to array
    y = Array(y); # view to array
    # functions to scale data to lie in [-3π, 3π] × [-3π, 3π] and revert that scaling back to original 
    min_x, max_x = extrema(x)
    min_y, max_y = extrema(y)

    # scaling(x) = eltype(x)(2π+0) * (x .- minimum(x)) / (maximum(x) - minimum(x)) # we use [0, 2π] as our range
    x = eltype(x)(2π+0) * (x .- min_x) / (max_x - min_x)
    y = eltype(y)(2π+0) * (y .- min_y) / (max_y - min_y)

    k_x = range(FT1(-ms/2), stop=FT1((ms-1)/2), length=ms) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    k_y = range(FT1(-mt/2), stop=FT1((mt-1)/2), length=mt) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1



    # scale our wavenumbers to get back the original wavenumbers
    k_x = k_x .* FT1(2π) / (max_x - min_x) # can't assign in place to range
    k_y = k_y .* FT1(2π) / (max_y - min_y)



    # after scaling k, we have 0 to max_x-min_x and need to shift to min_x to max_x.., so we shift right by min_x (this doesnt change the amplitude and thus the energy spectrum...)
    # shift_x = exp.(-1im * k_x * min_x)
    # shift_y = exp.(-1im * k_y * min_y)
    shift = exp.(-1im * k_x * min_x) .* exp.(-1im * k_y * min_y)' # the fft of the shift in real space... (y should be transposed?)


    nufft_KEs = zeros(Complex{FT2}, ms, mt ,1 , NTT)

    # valid::MVector{length(x), Bool} = trues(size(x))# could use a MVector (incredibly slow for precompilatino which is bad bc length(x_vecs) changes all the time if we filter out nans in python...
    valid::BitVector = trues(size(x))# could use a MVector (incredibly slow for precompilatino which is bad bc length(x_vecs) changes all the time if we filter out nans in python...

    for k in Base.OneTo(NTT::Int64)
        KE = view(KE_vec, :, k, 1)
        # drop nans (bc we can only drop points that are always nan, but many points are nan sometimes
        valid .= .!isnan.(KE)

        # short circuit if all are nan
        if sum(valid) == 0 # sum is allegedly the fastest
            nufft_KEs[:, :, 1, k] .= Complex{FT2}(NaN)
        else

            KE = Array{Complex{eltype(u)}}(KE); # view to array
        
            # NUFFT object
            nufft_KE = FINUFFT.nufft2d1(x[valid], y[valid], KE[valid], iflag, eps, ms, ms) # would use views but FINUFFT doesn't like it

            nufft_KE .*= shift # this is an ms x mt x 1 array

            # Return result
            nufft_KEs[:, :, 1, k] .= nufft_KE
        end

        
    end
    return nufft_KEs::Array{Complex{FT2},4}, k_x, k_y # output was type unstable I guess?
    # return nufft_us::Array{Complex{FT2},3}, nufft_v::Array{Complex{FT2},3}, k_x, k_y # output was type unstable I guess?
end