"""
GPU-accelerated structure function kernels using KernelAbstractions.jl.

This extension is loaded automatically when `KernelAbstractions` is loaded by the user.
The `gpu_calculate_structure_function` entry point accepts any KA-compatible backend:
  - `KernelAbstractions.CPU()` – for CPU-parallel testing / parity verification
  - `CUDABackend()` from CUDA.jl – for NVIDIA GPU acceleration
  - `ROCBackend()` from AMDGPU.jl – for AMD GPU acceleration

The kernel is O(N²) pair-wise and relies on atomic accumulation into distance bins.

!!! note "KernelAbstractions Macro Limitations"
    We explicitly import `@index` and `@atomic` from `KernelAbstractions` because 
    these macros currently fail to resolve correctly when called as `KA.@index` or `KA.@atomic`.
    This is a known issue in KernelAbstractions.jl (see https://github.com/JuliaGPU/KernelAbstractions.jl/issues/542#issuecomment-3986160961)
    and is expected to be fixed in a future release.
"""
module StructureFunctionsGPUCalculationsExt

using KernelAbstractions: KA, @index, @atomic
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, SpectralAnalysis as SFSA, HelperFunctions as SFH, StructureFunctionTypes as SFT

# ---------------------------------------------------------------------------
# Inner kernel
# ---------------------------------------------------------------------------

KA.@kernel function _sf_kernel!(
    output,                      # Vector{FT} of length N_bins-1
    counts,                      # Vector{FT} of length N_bins-1
    x_mat,                       # Matrix{FT} of size (N_dims, N_points)
    u_mat,                       # Matrix{FT} of size (N_dims, N_points)
    @Const(distance_bins),       # monotone bin edges, length N_bins
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
)
    # IMPORTANT: Use bare `@index` (not `KA.@index`) — see module-level note.
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        # Build SVector on-stack  (N_dims must be small, typically 1–3)
        X1 = SA.SVector{3}(x_mat[1,i], x_mat[2,i], x_mat[3,i])
        X2 = SA.SVector{3}(x_mat[1,j], x_mat[2,j], x_mat[3,j])
        U1 = SA.SVector{3}(u_mat[1,i], u_mat[2,i], u_mat[3,i])
        U2 = SA.SVector{3}(u_mat[1,j], u_mat[2,j], u_mat[3,j])

        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)

        # Binary search for bin index
        bin = SFH.digitize(dist, distance_bins)

        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            # IMPORTANT: Use bare `@atomic` (not `KA.@atomic`) — same reason as @index above.
            @atomic output[bin] += val
            @atomic counts[bin] += one(eltype(output))
        end
    end
end


# ---------------------------------------------------------------------------
# Spectral Analysis Kernel (Direct Sum)
# ---------------------------------------------------------------------------

KA.@kernel function _spectral_kernel!(
    coeffs,                # Out: (ms..., NU)
    @Const(x_dev),         # (3, N)  - padded to 3
    @Const(u_dev),         # (NU, N)
    @Const(ks_phys_dev),   # 3 vectors of varying lengths ms[d] - padded to 3
    iflag::Int,
    N::Int,
    NU::Int,
    D::Int,
    ms::NTuple # Use NTuple (untapped) to avoid UndefVarError
)
    # One thread per wavenumber I
    idx = @index(Global, Cartesian)
    
    # Pre-fetch k_phys components for this wavenumber
    # We use SVector{3} and dot with padded x_pos
    k_phys = SA.SVector{3, eltype(x_dev)}(
        D >= 1 ? ks_phys_dev[1][idx[1]] : zero(eltype(x_dev)),
        D >= 2 ? ks_phys_dev[2][idx[2]] : zero(eltype(x_dev)),
        D >= 3 ? ks_phys_dev[3][idx[3]] : zero(eltype(x_dev))
    )
    
    for u_idx in 1:NU
        sum_val = zero(eltype(coeffs))
        for j in 1:N
            x_pos = SA.SVector{3, eltype(x_dev)}(
                x_dev[1, j],
                x_dev[2, j],
                x_dev[3, j]
            )
            
            # Phase factor
            phi = -iflag * (SA.dot(k_phys, x_pos))
            W = complex(cos(phi), sin(phi)) 
            
            sum_val += u_dev[u_idx, j] * W
        end
        coeffs[idx, u_idx] = sum_val / N
    end
end


# ---------------------------------------------------------------------------
# N-dimensional variant: pads 1D/2D inputs to 3D for uniformity
# ---------------------------------------------------------------------------

function _pad3(v::SA.SVector{N, T}) where {N, T}
    if N == 1
        return SA.SVector{3, T}(v[1], zero(T), zero(T))
    elseif N == 2
        return SA.SVector{3, T}(v[1], v[2], zero(T))
    else
        return v
    end
end


# ---------------------------------------------------------------------------
# Public API – extends the stub declared in Calculations.jl
# ---------------------------------------------------------------------------

"""
    gpu_calculate_structure_function(backend, x_mat, u_mat, distance_bins, sf_type; workgroup_size=64)

Compute structure functions on `backend` (any KernelAbstractions backend).

# Arguments
- `backend`: e.g. `KernelAbstractions.CPU()`, `CUDA.CUDABackend()`, etc.
- `x_mat`: `(N_dims, N_points)` matrix of spatial positions.
- `u_mat`: `(N_dims, N_points)` matrix of velocity components.
- `distance_bins`: monotone vector of bin *edges* (length = N_bins).
- `sf_type`: any `AbstractStructureFunctionType`.

# Returns
`(sf_values, counts)` – both are host-side `Vector{Float64}` of length `N_bins - 1`.
"""
function SFC.gpu_calculate_structure_function(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    sf_type::SFT.AbstractStructureFunctionType;
    workgroup_size::Int = 64,
) where {FT}
    N_dims, N_points = size(x_mat)
    N_bins = length(distance_bins)

    if N_dims > 3
        error("GPUCalculationsExt: only 1D–3D inputs are supported (got N_dims=$N_dims)")
    end

    # Pad to 3D so the kernel can use fixed-size SVectors
    x3 = zeros(FT, 3, N_points)
    u3 = zeros(FT, 3, N_points)
    x3[1:N_dims, :] .= x_mat
    u3[1:N_dims, :] .= u_mat

    # Allocate device arrays
    x_dev    = KA.allocate(backend, FT, 3, N_points)
    u_dev    = KA.allocate(backend, FT, 3, N_points)
    bins_dev = KA.allocate(backend, FT, N_bins)
    out_dev  = KA.zeros(backend, FT, N_bins - 1)
    cnt_dev  = KA.zeros(backend, FT, N_bins - 1)

    copyto!(x_dev, x3)
    copyto!(u_dev, u3)
    copyto!(bins_dev, collect(distance_bins))

    n_threads = N_points * N_points

    kernel! = _sf_kernel!(backend, workgroup_size)
    kernel!(
        out_dev, cnt_dev,
        x_dev, u_dev,
        bins_dev,
        sf_type,
        N_points, N_dims, N_bins;
        ndrange = (N_points, N_points),
    )
    KA.synchronize(backend)

    return Array(out_dev), Array(cnt_dev)
end


# ---------------------------------------------------------------------------
# Spectral Analysis API Extension
# ---------------------------------------------------------------------------

function SFSA.gpu_calculate_spectrum(
    backend::KA.Backend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::NTuple{D, Int};
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    workgroup_size::Int = 16
) where {D}
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])
    NU = length(u_vecs)

    # 1. Coordinate ranges (replicated from SpectralAnalysis.jl)
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(x_vecs[d])
            return max_x - min_x
        end
    end
    
    ks_phys = ntuple(d -> range(FT(-ms[d]÷2), stop=FT((ms[d]-1)÷2), length=ms[d]) .* (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])), Val(D))

    # 2. Allocate and transfer
    # Standardize to 3D padding for SVector compatibility in kernel
    x_mat = zeros(FT, 3, N)
    u_mat = zeros(FT, NU, N)
    for d in 1:D; x_mat[d, :] .= x_vecs[d]; end
    for u_idx in 1:NU; u_mat[u_idx, :] .= u_vecs[u_idx]; end

    x_dev = KA.allocate(backend, FT, 3, N)
    u_dev = KA.allocate(backend, FT, NU, N)
    copyto!(x_dev, x_mat)
    copyto!(u_dev, u_mat)

    # Transfer ks_phys as vectors, padded to 3
    ks_phys_dev = ntuple(d_ -> begin
        if d_ <= D
            v = KA.allocate(backend, FT, length(ks_phys[d_]))
            copyto!(v, collect(ks_phys[d_]))
            return v
        else
            # Dummy for padding
            return KA.allocate(backend, FT, 1)
        end
    end, Val(3))

    coeffs_dev = KA.zeros(backend, Complex{FT}, ms..., NU)

    # 3. Launch kernel
    # ndrange is the grid of wavenumbers ms...
    kernel! = _spectral_kernel!(backend, workgroup_size)
    kernel!(
        coeffs_dev,
        x_dev, u_dev,
        ks_phys_dev,
        iflag,
        N, NU, D,
        ms;
        ndrange = ms
    )
    KA.synchronize(backend)

    return Array(coeffs_dev), ks_phys
end


end # module GPUCalculationsExt