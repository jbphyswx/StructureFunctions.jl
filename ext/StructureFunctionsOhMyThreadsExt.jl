module StructureFunctionsOhMyThreadsExt

using Distances: Distances as DI
using OhMyThreads: OhMyThreads as OMT
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using StructureFunctions:
    StructureFunctions as SF,
    Calculations as SFC,
    StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT,
    HelperFunctions as SFH

# --- 1D Tuple thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT}
    if verbose
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    # Chunked tmapreduce: O(n_tasks) allocations instead of O(N_points)
    result = OMT.tmapreduce(+, OMT.chunks(eachindex(x_vecs[1]); n=Threads.nthreads())) do chunk
        local_output = zeros(OT, length(distance_bins))
        local_counts = zeros(OT, length(distance_bins))
        bin_edges = SFC.flat_bin_edges(distance_bins)
        vN = Val(length(x_vecs))
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                vN,
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                bin_edges;
                distance_metric = distance_metric,
            )
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_output, local_counts)
    end
    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    SFC.threaded_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )

    if RSAC
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    end

    counts_safe = copy(counts)
    counts_safe[counts_safe .== 0] .= NaN
    output_div = output ./ counts_safe
    return SF.StructureFunction(structure_function_type, distance_bins, output_div)
end

# --- 1D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    if verbose
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    N3 = length(distance_bins)
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    N = size(x_arr, 1)
    if !(N in (1, 2, 3))
        throw(ArgumentError("Threaded array backend supports only 1D, 2D, or 3D inputs."))
    end

    # Chunked tmapreduce to allocate once per thread/task chunk
    result = OMT.tmapreduce(+, OMT.chunks(axes(x_arr, 2); n=Threads.nthreads())) do chunk
        local_output = zeros(OT, N3)
        local_counts = zeros(OT, N3)
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                Val(N),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_output, local_counts)
    end

    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    SFC.threaded_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        kwargs...,
    )

    if RSAC
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    end

    counts_safe = copy(counts)
    counts_safe[counts_safe .== 0] .= NaN
    output_div = output ./ counts_safe
    return SF.StructureFunction(structure_function_type, distance_bins, output_div)
end

# --- 2D Tuple thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT}
    if verbose
        @info("calculating 2D joint structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    distance_bins_vec = SFC.flat_bin_edges(distance_bins)
    value_bins_vec = SFC.flat_bin_edges(value_bins)
    N3 = length(distance_bins)
    N4 = length(value_bins_vec) - 1

    # Chunked tmapreduce: O(n_tasks) allocations instead of O(N_points)
    result = OMT.tmapreduce(+, OMT.chunks(eachindex(x_vecs[1]); n=Threads.nthreads())) do chunk
        local_sums = zeros(OT, N3, N4)
        local_counts = zeros(OT, N3, N4)
        vN = Val(length(x_vecs))
        for i in chunk
            SFC.calculate_structure_function_2d_i!(
                local_sums,
                local_counts,
                vN,
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                distance_bins_vec,
                value_bins_vec;
                distance_metric = distance_metric,
            )
        end
        SFO.StructureFunction2D(
            structure_function_type,
            distance_bins,
            value_bins,
            local_sums,
            local_counts,
        )
    end

    sums_2d .+= result.sums
    counts_2d .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
)
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    value_bins_vec = SFC.flat_bin_edges(value_bins)
    N4 = length(value_bins_vec) - 1

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(OT, N3, N4)

    SFC.threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
    )
end

# --- 2D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC.threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    value_bins_vec = SFC.flat_bin_edges(value_bins)
    N4 = length(value_bins_vec) - 1

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(OT, N3, N4)

    SFC.threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        value_bins;
        kwargs...,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
    )
end

# --- Preservation of the existing _dispatch_single_pass method ---

function SFC._dispatch_single_pass(
    ::SF.ThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    thread_sums = nothing,
    thread_counts = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_threads = Threads.nthreads()
    
    # Check/allocate thread-local reduction heaps
    ts = isnothing(thread_sums) ? zeros(OT, 8, n_bins, n_threads) : thread_sums
    tc = isnothing(thread_counts) ? zeros(Int64, 8, n_bins, n_threads) : thread_counts
    
    fill!(ts, zero(OT))
    fill!(tc, 0)
    
    n_points = size(x, 2)
    
    OMT.tforeach(1:n_points) do i
        tid = Threads.threadid()
        x_i = SA.SVector{2, FT1}(x[1, i], x[2, i])
        u_i = SA.SVector{2, FT2}(u[1, i], u[2, i])
        
        for j in (i+1):n_points
            x_j = SA.SVector{2, FT1}(x[1, j], x[2, j])
            
            dx = SFH.δr(x_i, x_j)
            r = LA.norm(dx)
            
            bin_idx = SFH.digitize(r, distance_bins)
            
            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{2, FT2}(u[1, j], u[2, j])
                du = u_j - u_i
                
                rh = SFH.r̂(x_i, x_j)
                nh = SFH.n̂(rh)
                
                du_L = LA.dot(du, rh)
                du_T = LA.dot(du, nh)
                
                du_L2 = du_L * du_L
                du_T2 = du_T * du_T
                
                @inbounds ts[1, bin_idx, tid] += du_L2 + du_T2
                @inbounds ts[2, bin_idx, tid] += du_L2
                @inbounds ts[3, bin_idx, tid] += du_T2
                @inbounds ts[4, bin_idx, tid] += du_L * (du_L2 + du_T2)
                @inbounds ts[5, bin_idx, tid] += du_L * du_L2
                @inbounds ts[6, bin_idx, tid] += du_L2 * du_T
                @inbounds ts[7, bin_idx, tid] += du_L * du_T2
                @inbounds ts[8, bin_idx, tid] += du_T * du_T2
                
                for t in 1:8
                    @inbounds tc[t, bin_idx, tid] += 1
                end
            end
        end
    end
    
    # Reduce thread-local slices
    sums = zeros(OT, 8, n_bins)
    counts = zeros(Int64, 8, n_bins)
    for tid in 1:n_threads
        for k in 1:n_bins
            for t in 1:8
                sums[t, k] += ts[t, k, tid]
                counts[t, k] += tc[t, k, tid]
            end
        end
    end
    
    return SFC.postprocess_single_pass_results(sums, counts, distance_bins)
end

end # module
