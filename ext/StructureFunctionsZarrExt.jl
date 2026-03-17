module StructureFunctionsZarrExt

using Zarr: Zarr
using DiskArrays: DiskArrays
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH

"""
    calculate_structure_function_from_file(::Val{:zarr}, fpath::String, bin_edges, sf_type; 
                                          x_key=("x1", "x2"), u_key=("u1", "u2"), kwargs...)

Load data from a Zarr store and calculate the structure function.
Supports out-of-core computation if the backend supports it (to be enhanced in I2).
"""
function SFC.calculate_structure_function_from_file(
    ::Val{:zarr},
    fpath::String,
    bin_edges,
    sf_type;
    x_key = ("x1", "x2"),
    u_key = ("u1", "u2"),
    kwargs...
)
    # Open Zarr group
    z = Zarr.zopen(fpath, "r")
    
    # 1. Load data handles (lazy)
    x_raw = _load_zvars(z, x_key)
    u_raw = _load_zvars(z, u_key)

    # 2. For now, we materialise (Block I2 will enable lazy processing)
    # To enable true out-of-core, core calculations need to support AbstractDiskArray.
    x_flat = SFH.flatten_data(x_raw)
    u_flat = SFH.flatten_data(u_raw)

    # Materialize (Block I2 TODO: lazy chunks)
    FT = eltype(u_flat[1])
    D = length(x_flat)
    NU = length(u_flat)
    N = length(u_flat[1])

    x_mat = zeros(FT, D, N)
    u_mat = zeros(FT, NU, N)
    for i in 1:D; x_mat[i, :] .= x_flat[i][:]; end
    for i in 1:NU; u_mat[i, :] .= u_flat[i][:]; end

    # 3. Clean NaNs
    x_clean, u_clean = SFH.remove_nans(x_mat, u_mat)

    # 4. Delegate
    return SFC.calculate_structure_function(sf_type, x_clean, u_clean, bin_edges; kwargs...)
end

function _load_zvars(z, key)
    if key isa Union{Tuple, AbstractVector}
        return ntuple(i -> z[string(key[i])], length(key))
    else
        return (z[string(key)],)
    end
end

end # module
