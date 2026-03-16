# I wanna test if I can get FINUFFT to take a 2D NUFFT

using FINUFFT # see https://github.com/ludvigak/FINUFFT.jl, https://ludvigak.github.io/FINUFFT.jl/latest, https://finufft.readthedocs.io/en/latest/math.html 
using StaticArrays

# 2D NUFFT type 1

function test_2d_nufft(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{NTuple{N2, FT2}, SVector{N2, FT2}, AbstractVector{FT2}}}; # Tuple{Vararg{Vector{FT},N}}
    ms::Int = 128,
    mt::Int = 128,
) where {N, N2, FT1, FT2}
    """
    Test 2D NUFFT
    - using type1 so we get unifrom outputs (i think that's faster and/or more accurate?)
    """

    x, y = eachrow( reduce( hcat, (map(Array, x_vecs)))); # this line is slow bc of precompilation, precompiles for every different length of x_vecs because why?

    x = Array(x); # view to array
    y = Array(y); # view to array

    # functions to scale data to lie in [-3π, 3π] × [-3π, 3π] and revert that scaling back to original 
    min_x, max_x = extrema(x)
    min_y, max_y = extrema(y)

    # scaling(x) = eltype(x)(2π+0) * (x .- minimum(x)) / (maximum(x) - minimum(x)) # we use [0, 2π] as our range
    x = eltype(x)(2π+0) * (x .- min_x) / (max_x - min_x)
    y = eltype(y)(2π+0) * (y .- min_y) / (max_y - min_y)

    u, v = eachrow( reduce(hcat, (map(Array, u_vecs))));
    u = Array{Complex{eltype(u)}}(u); # view to array
    v = Array{Complex{eltype(v)}}(v); # view to array

    # NUFFT object
    nufft_u = FINUFFT.nufft2d1(x, y, u, iflag, eps, ms, ms)
    nufft_v = FINUFFT.nufft2d1(x, y, v, iflag, eps, ms, mt)

    k_x = range(FT1(-ms/2), stop=FT1((ms-1)/2), length=ms) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    k_y = range(FT1(-mt/2), stop=FT1((mt-1)/2), length=mt) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    # scale our wavenumbers to get back the original wavenumbers
    k_x .* FT1(2π) / (max_x - min_x)
    k_y .* FT1(2π) / (max_y - min_y)

    # after scaling k, we have 0 to max_x-min_x and need to shift to min_x to max_x.., so we shift right by min_x (this doesnt change the amplitude and thus the energy spectrum...)
    shift_x = exp.(-1im * k_x * min_x)
    shift_y = exp.(-1im * k_y * min_y)

    nufft_u .*= shift_x
    nufft_v .*= shift_y

    # Return result
    return  nufft_u::Array{Complex{FT2},3}, nufft_v::Array{Complex{FT2},3}, k_x, k_y # output was type unstable I guess?
end


function test_2d_nufft(
    x_vecs::AbstractVector{ <:Union{NTuple{N2, FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # testing if vector avoids repeated compilation.... doesn't seem to lol... idk
    u_vecs::AbstractVector{ <:Union{NTuple{N2, FT2}, SVector{N2, FT2}, AbstractVector{FT2}}}; # Tuple{Vararg{Vector{FT},N}}
    ms::Int64 = 128,
    mt::Int64 = 128,
    iflag::Int64 = 1,
    eps::FT1 = FT1(1e-6),
) where {N2 <:Int64, FT1 <: AbstractFloat, FT2 <: AbstractFloat}
    """
    Test 2D NUFFT
    - using type1 so we get unifrom outputs (i think that's faster and/or more accurate?)
    """
    x, y = eachrow( reduce( hcat, (map(Array, x_vecs)))); # this line is slow bc of precompilation, precompiles for every different length of x_vecs because why?

    x = Array(x); # view to array
    y = Array(y); # view to array

    # functions to scale data to lie in [-3π, 3π] × [-3π, 3π] and revert that scaling back to original 
    min_x, max_x = extrema(x)
    min_y, max_y = extrema(y)

    # scaling(x) = eltype(x)(2π+0) * (x .- minimum(x)) / (maximum(x) - minimum(x)) # we use [0, 2π] as our range
    x = eltype(x)(2π+0) * (x .- min_x) / (max_x - min_x)
    y = eltype(y)(2π+0) * (y .- min_y) / (max_y - min_y)

    u, v = eachrow( reduce(hcat, (map(Array, u_vecs))));
    u = Array{Complex{eltype(u)}}(u); # view to array
    v = Array{Complex{eltype(v)}}(v); # view to array
 
    # NUFFT object
    nufft_u = FINUFFT.nufft2d1(x, y, u, iflag, eps, ms, ms)
    nufft_v = FINUFFT.nufft2d1(x, y, v, iflag, eps, ms, mt)

    k_x = range(FT1(-ms/2), stop=FT1((ms-1)/2), length=ms) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    k_y = range(FT1(-mt/2), stop=FT1((mt-1)/2), length=mt) # k from the mode range, which is integers lying in [-m/2, (m-1)/2] , see https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1
    # scale our wavenumbers to get back the original wavenumbers
    k_x .* FT1(2π) / (max_x - min_x)
    k_y .* FT1(2π) / (max_y - min_y)

    # after scaling k, we have 0 to max_x-min_x and need to shift to min_x to max_x.., so we shift right by min_x (this doesnt change the amplitude and thus the energy spectrum...)
    shift_x = exp.(-1im * k_x * min_x)
    shift_y = exp.(-1im * k_y * min_y)

    nufft_u .*= shift_x
    nufft_v .*= shift_y

    # Return result
    return nufft_u::Array{Complex{FT2},3}, nufft_v::Array{Complex{FT2},3}, k_x, k_y # output was type unstable I guess?
end