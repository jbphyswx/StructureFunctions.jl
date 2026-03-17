"""
GPU-accelerated structure function kernels using KernelAbstractions.jl.

This extension is loaded automatically when `KernelAbstractions` is loaded by the user.
The `gpu_calculate_structure_function` entry point accepts any KA-compatible backend:
  - `KernelAbstractions.CPU()` – for CPU-parallel testing / parity verification
  - `CUDABackend()` from CUDA.jl – for NVIDIA GPU acceleration
  - `ROCBackend()` from AMDGPU.jl – for AMD GPU acceleration

The kernel is O(N²) pair-wise and relies on atomic accumulation into distance bins.
"""
module GPUCalculationsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic
#
# IMPORTANT: Do NOT use `KA.@index` or `KA.@atomic` inside `@kernel` functions.
# The `@KA.kernel` macro rewrites bare `@index` and `@atomic` symbols in the
# AST to inject hidden thread/block context arguments (like `__ctx__`). When
# these macros are written with a module prefix (`KA.@index`), the parser sees
# a different pattern and skips the rewrite, so at runtime the functions are
# called without the required context argument, causing a MethodError.
# Fix: import them bare and use the bare form inside every @kernel body.
#
# Upstream tracking issue (fix expected in a future KA release):
# https://github.com/JuliaGPU/KernelAbstractions.jl/issues/542#issuecomment-3986160961
#
# TODO: Once the upstream fix is released and we bump our KA compat bound,
#       this workaround (and this comment) can be removed.
using StaticArrays: StaticArrays as SA
import Distances: Distances as DI
import StructureFunctions as SF

const Calculations = SF.Calculations
const HF = SF.HelperFunctions
const SFT = SF.StructureFunctionTypes

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
        bin = HF.digitize(dist, distance_bins)

        if 1 <= bin < N_bins
            r̂ = HF.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            # IMPORTANT: Use bare `@atomic` (not `KA.@atomic`) — same reason as @index above.
            @atomic output[bin] += val
            @atomic counts[bin] += one(eltype(output))
        end
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
function Calculations.gpu_calculate_structure_function(
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

end # module GPUCalculationsExt