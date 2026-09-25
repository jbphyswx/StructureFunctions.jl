# Exact CUDA culling preparation. Coordinates, the sort permutation, and the sorted field stay on
# the device. Only occupied cell ids and their run starts are copied to the host; those compact
# arrays drive the small tile-pair schedule builder shared by every backend.

function _cuda_cull_cell_ids!(cell_ids, x, origin, inv_h, dims, n::Int, ::Val{D}) where {D}
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    if i <= n
        linear = Int64(0)
        stride = Int64(1)
        @inbounds for d in 1:D
            c = 1 + floor(Int, (x[d, i] - origin[d]) * inv_h)
            c = min(max(c, 1), dims[d])
            linear += Int64(c - 1) * stride
            stride *= Int64(dims[d])
        end
        @inbounds cell_ids[i] = linear + 1
    end
    return nothing
end

function _cuda_cull_run_flags!(flags, sorted_ids, n::Int)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    if i <= n
        @inbounds flags[i] = i == 1 || sorted_ids[i] != sorted_ids[i - 1]
    end
    return nothing
end

function SFC.gpu_device_cull_grid(::CUDA.CUDABackend, x::CUDA.CuArray{FT, 2}, cutoff,
                                  policy::SFC.CullingPolicy) where {FT}
    d, n = size(x)
    n >= 2 || return nothing
    d in (2, 3) || return nothing
    dimension = d == 2 ? Val(2) : Val(3)

    origin = ntuple(i -> FT(minimum(view(x, i, :))), dimension)
    hi = ntuple(i -> FT(maximum(view(x, i, :))), dimension)
    span = SFC.SF_CULL_CELLS_PER_CUTOFF
    cutoff_ft = FT(cutoff)
    inv_h = inv(cutoff_ft / span)
    dims = ntuple(i -> max(1, floor(Int, (hi[i] - origin[i]) * inv_h) + 1), dimension)
    SFC._cull_is_worthwhile(policy, dims, span) || return nothing

    cell_ids = CUDA.CuArray{Int64}(undef, n)
    threads = min(256, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cuda_cull_cell_ids!(
        cell_ids, x, origin, inv_h, dims, n, dimension)

    perm = sortperm(cell_ids)
    sorted_ids = cell_ids[perm]
    flags = CUDA.CuArray{Bool}(undef, n)
    @cuda threads=threads blocks=blocks _cuda_cull_run_flags!(flags, sorted_ids, n)
    run_starts_device = findall(flags)
    occupied_ids = Array(sorted_ids[run_starts_device])
    run_starts = Array(run_starts_device)
    push!(run_starts, n + 1)
    offsets = SFC.cull_row_offsets(span, dimension)
    return SFC.CellGrid(
        origin, inv_h, dims, occupied_ids, run_starts, perm, offsets, cutoff_ft, span,
    )
end
