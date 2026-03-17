"""
The workhorse of this package
"""
module Calculations
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using ..HelperFunctions: HelperFunctions as SFH
using ..StructureFunctionTypes: StructureFunctionTypes as SFT
using ..StructureFunctionObjects: StructureFunctionObjects as SFO

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Base.Threads: Threads
using LoopVectorization: LoopVectorization as LV # TODO: Move to extension or replace with Polyester/SIMD as part of modernization

export calculate_structure_function, parallel_calculate_structure_function, gpu_calculate_structure_function, calculate_structure_function_from_file

"""
    calculate_structure_function(sf_type::SFT.AbstractStructureFunctionType, path::String, bins; kwargs...)

Entry point for calculating structure functions from a file (e.g. NetCDF, JLD2, CSV).
The file extension is used to dispatch to an appropriate extension method.
"""
function calculate_structure_function(sf_type::SFT.AbstractStructureFunctionType, path::String, bins; kwargs...)
    ext_str = lowercase(split(path, '.')[end])
    return calculate_structure_function_from_file(Val(Symbol(ext_str)), path, bins, sf_type; kwargs...)
end

"""
    calculate_structure_function(sf_sym::Symbol, args...; kwargs...)

Shorthand that looks up the operator by name and calls the core calculation.
"""
function calculate_structure_function(sf_sym::Symbol, args...; kwargs...)
    return calculate_structure_function(Val(sf_sym), args...; kwargs...)
end

function calculate_structure_function(::Val{sf_sym}, args...; kwargs...) where {sf_sym}
    return calculate_structure_function(SFT.get_structure_function_type(Val(sf_sym)), args...; kwargs...)
end

"""
    calculate_structure_function(order::Int, mode::Symbol, args...; kwargs...)

Shorthand that looks up the operator by order and mode and calls the core calculation.
"""
function calculate_structure_function(order::Int, mode::Symbol, args...; kwargs...)
    return calculate_structure_function(Val(order), Val(mode), args...; kwargs...)
end

function calculate_structure_function(::Val{order}, ::Val{mode}, args...; kwargs...) where {order, mode}
    return calculate_structure_function(SFT.get_structure_function_type(Val(order), Val(mode)), args...; kwargs...)
end

# Shorthand for parallel version too
function parallel_calculate_structure_function(sf_sym::Symbol, args...; kwargs...)
    return parallel_calculate_structure_function(Val(sf_sym), args...; kwargs...)
end

function parallel_calculate_structure_function(::Val{sf_sym}, args...; kwargs...) where {sf_sym}
    return parallel_calculate_structure_function(SFT.get_structure_function_type(Val(sf_sym)), args...; kwargs...)
end

function parallel_calculate_structure_function(order::Int, mode::Symbol, args...; kwargs...)
    return parallel_calculate_structure_function(Val(order), Val(mode), args...; kwargs...)
end

function parallel_calculate_structure_function(::Val{order}, ::Val{mode}, args...; kwargs...) where {order, mode}
    return parallel_calculate_structure_function(SFT.get_structure_function_type(Val(order), Val(mode)), args...; kwargs...)
end

"""
    StructureFunction(order::Int, mode::Symbol, x, u, bins; kwargs...)

Factory constructor that calculates a structure function and returns a result object.
This redirects to `calculate_structure_function`.
"""
function SFO.StructureFunction(order::Int, mode::Symbol, args...; kwargs...)
    return calculate_structure_function(Val(order), Val(mode), args...; kwargs...)
end

function SFO.StructureFunction(::Val{order}, ::Val{mode}, args...; kwargs...) where {order, mode}
    return calculate_structure_function(Val(order), Val(mode), args...; kwargs...)
end

function SFO.StructureFunction(sf_sym::Symbol, args...; kwargs...)
    return calculate_structure_function(Val(sf_sym), args...; kwargs...)
end

function SFO.StructureFunction(::Val{sf_sym}, args...; kwargs...) where {sf_sym}
    return calculate_structure_function(Val(sf_sym), args...; kwargs...)
end

"""
    gpu_calculate_structure_function(...)

GPU-accelerated structure function calculation. Requires loading `KernelAbstractions.jl`
to activate the `GPUCalculationsExt` extension. The backend can be `KernelAbstractions.CPU()`
(for testing parity) or any GPU backend like `CUDABackend()` from `CUDA.jl`.

This stub exists so the extension can legally extend this function.
"""
function gpu_calculate_structure_function end

# Stub for file extensions
function calculate_structure_function_from_file end

# ---------------------------------------------------------------------------
# Functor Support: Operator(x, u, bins; kwargs...) -> calculate_structure_function
# ---------------------------------------------------------------------------

function (sf::SFT.AbstractStructureFunctionType)(x, u, bins; kwargs...)
    return calculate_structure_function(sf, x, u, bins; kwargs...)
end

function (sf::SFT.AbstractStructureFunctionType)(x, u, bins, ::Val{RSAC}; kwargs...) where {RSAC}
    return calculate_structure_function(sf, x, u, bins, Val(RSAC); kwargs...)
end

function (::Type{T})(x, u, bins; kwargs...) where {T <: SFT.AbstractStructureFunctionType}
    return calculate_structure_function(T(), x, u, bins; kwargs...)
end

function (::Type{T})(x, u, bins, ::Val{RSAC}; kwargs...) where {T <: SFT.AbstractStructureFunctionType, RSAC}
    return calculate_structure_function(T(), x, u, bins, Val(RSAC); kwargs...)
end

"""
    calculate_structure_function(sf_type, x, u, bin_edges::AbstractVector{<:Number}, ...)

Convenience method that converts a vector of bin edges `[e1, e2, e3]` into 
adjacent bins `[(e1, e2), (e2, e3)]` and calls the core calculation.
"""
function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bin_edges::AbstractVector{<:Number},
    args...;
    kwargs...
)
    bin_tuples = [(bin_edges[i], bin_edges[i+1]) for i in 1:(length(bin_edges)-1)]
    return calculate_structure_function(structure_function_type, x, u, bin_tuples, args...; kwargs...)
end

###########
# Core Calculation Methods
########

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}};
    return_sums_and_counts = Val(false),
    kwargs...,
) where {T1, T2, FT3}
    return calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        return_sums_and_counts isa Bool ? Val(return_sums_and_counts) : return_sums_and_counts;
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {T1, T2, FT3, RSAC}
    return _calculate_structure_function_core(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

function _calculate_structure_function_core(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {T1, T2, FT3, RSAC}
    N = length(x_vecs)
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # preallocate output as vector of length of distance_bins
    # Use promote_type and float to ensure output can hold float results/NaN
    OT = promote_type(float(FT1), float(FT2)) 
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    # Create a stable Vector for bins edges
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    if verbose
        @info("calculating structure function")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
    lock = Threads.ReentrantLock()
    Threads.@threads for i::Int64 in iter_inds
        _output, _counts = calculate_structure_function_i(
            structure_function_type,
            i,
            x_vecs,
            u_vecs,
            distance_bins_vec;
            distance_metric = distance_metric,
        )
        Threads.lock(lock) do
            output .+= _output
            counts .+= _counts
        end
    end

    if RSAC # just return the sums and the counts, don't take the mean in each bin...
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, output, counts)
    else # do the mean in each bin.
        counts_safe = copy(counts) # copy to avoid mutating the original counts
        counts_safe[counts_safe .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output_div = output ./ counts_safe
        return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end



function calculate_structure_function_i(
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins_vec::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {T1, T2, FT3}
    N = length(x_vecs)
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins_vec)



    # Commit A2: Fixed "double calculating" by iterating only over unique pairs (j > i).
    # Removed BadImplementationError as this path is now considered correct.

    # preallocate output as vector of length of distance_bins
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, N3 - 1)
    counts = zeros(OT, N3 - 1)

    iter_inds = eachindex(x_vecs[1]) 
    
    # PERFORMANCE NOTE: Converting to SVector at the loop boundary (hoisting)
    # is critical for zero heap-allocations in the inner loop. 
    # v2 - v1 on SubArrays/Vectors triggers heap temporaries via broadcasting.
    # SVector arithmetic is handled entirely on the stack/CPU registers.
    # We load U1/X1 once here to avoid redundant memory access in the j-loop.
    X1 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][i], Val{N}()))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val{N}()))
    
    # Iterate only over unique pairs where j > i to avoid double calculation
    for j in (i+1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][j], Val{N}()))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val{N}()))

        distance = distance_metric(X1, X2)
        bin = SFH.digitize(distance, distance_bins_vec)
        if 1 <= bin < N3
            @inbounds output[bin] += structure_function_type(U2 - U1, SFH.r̂(X1, X2))
            @inbounds counts[bin] += 1
        end
    end
    return output, counts
end




function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {T1, T2}
    N = length(x_vecs)
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_vecs, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing == :linear
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing ∈ (:logarithmic, :log)
        distance_bins = 10 .^ range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        error("bin_spacing must be :linear or :logarithmic/:log")
    end
    FT3 = eltype(distance_bins)
    distance_bins = SA.SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges
    return calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end


function minmax_i(
    i::Int,
    x_vecs::Tuple{T, Vararg{T}},
    distance_metric = DI.Euclidean(),
) where {T}
    N = length(x_vecs)
    FT = eltype(T)
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][i], Val{N}()))
    for j in eachindex(x_vecs[1])
        if i != j
            @inbounds X2 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][j], Val{N}()))
            # update the min and max distances
            distance = distance_metric(X1, X2) 
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end


### ====================================================== ###
### ====================================================== ###
### ====================================================== ###
"""
Array version (is slower lol)
"""
# array version seems to be slower lol, idk why (and we're even using views!)

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}};
    return_sums_and_counts = Val(false),
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    return calculate_structure_function(
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        return_sums_and_counts isa Bool ? Val(return_sums_and_counts) : return_sums_and_counts;
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    return _calculate_structure_function_core(
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

function _calculate_structure_function_core(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    N3 = length(distance_bins)
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3)
    counts = zeros(N3)

    # Create a stable Vector for bins edges
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    if verbose
        @info("calculating structure function")
    end

    iter_inds = axes(x_arr,2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    N = size(x_arr, 1)
    vN = if N == 1 Val{1}() elseif N == 2 Val{2}() elseif N == 3 Val{3}() else Val(N) end
        PM.@showprogress enabled = show_progress for i in iter_inds 
            _output, _counts = calculate_structure_function_i(
                vN,
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        output .+= _output
        counts .+= _counts
    end

    if RSAC # just return the sums and the counts, don't take the mean in each bin...
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, output, counts)
    else # do the mean in each bin.
        counts_safe = copy(counts)
        counts_safe[counts_safe .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output_div = output ./ counts_safe
        return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end

function calculate_structure_function_i(
    ::Val{N},
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins_vec::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {N, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N3 = length(distance_bins_vec)

    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3 - 1)
    counts = zeros(N3 - 1)

    iter_inds = axes(x_arr,2) 
    
    # PERFORMANCE NOTE: Converting to SVector at the loop boundary (hoisting)
    # is critical for zero heap-allocations in the inner loop. 
    # v2 - v1 on SubArrays/Vectors triggers heap temporaries via broadcasting.
    # SVector arithmetic is handled entirely on the stack/CPU registers.
    # We load U1/X1 once here to avoid redundant memory access in the j-loop.
    X1 = SA.SVector{N, FT1}(ntuple(k -> x_arr[k, i], Val{N}()))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_arr[k, i], Val{N}()))

    for j in iter_inds
        if i != j 
            X2 = SA.SVector{N, FT1}(ntuple(k -> x_arr[k, j], Val{N}()))
            U2 = SA.SVector{N, FT2}(ntuple(k -> u_arr[k, j], Val{N}()))

            distance = distance_metric(X1, X2) 
            bin = SFH.digitize(distance, distance_bins_vec) 
            if 1 <= bin < N3
                @inbounds output[bin] += structure_function_type(U2 - U1, SFH.r̂(X1, X2))
                @inbounds counts[bin] += 1
            end
        end
    end
    return output, counts
end



function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1 <: Number, FT2 <: Number}
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = axes(x_arr,2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_arr, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing == :linear
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing ∈ (:logarithmic, :log)
        distance_bins = 10 .^ range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        error("bin_spacing must be :linear or :logarithmic/:log")
    end
    FT3 = eltype(distance_bins)
    distance_bins = SA.SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges
    return calculate_structure_function(
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end

function minmax_i(
    i::Int,
    x_arr::AbstractArray{FT},
    distance_metric = DI.Euclidean(),
) where {FT <: Real}
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = @view(x_arr[:, i])
    # @inbounds X1 = @view(x_arr[i, :])

    for j in axes(x_arr,2)
    # for j in axes(x_arr,1)
        if i != j
            @inbounds X2 = @view(x_arr[:, j]) 
            # @inbounds X2 = @view(x_arr[j, :]) 

            # update the min and max distances
            distance = distance_metric(X1, X2) # this is the slow part
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end






































function parallel_calculate_structure_function end # placeholder for parallel extension

end
