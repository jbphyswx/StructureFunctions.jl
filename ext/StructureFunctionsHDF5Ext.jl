module StructureFunctionsHDF5Ext

using HDF5: HDF5
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH

"""
    calculate_structure_function_from_file(::Val{:h5}, fpath::String, bin_edges, sf_type; 
                                          x_key="x", u_key="u", kwargs...)

Load data from an HDF5 file and calculate the structure function.
"""
function SFC.calculate_structure_function_from_file(
    ::Val{:h5},
    fpath::String,
    bin_edges,
    sf_type;
    x_key = "x",
    u_key = "u",
    kwargs...
)
    HDF5.h5open(fpath, "r") do file
        # 1. Load data
        x_raw = _load_h5vars(file, x_key)
        u_raw = _load_h5vars(file, u_key)

        # 2. Flatten
        x_flat = SFH.flatten_data(x_raw)
        u_flat = SFH.flatten_data(u_raw)

        # 3. Convert to matrices
        FT = eltype(u_flat[1])
        D = length(x_flat)
        NU = length(u_flat)
        N = length(u_flat[1])

        x_mat = zeros(FT, D, N)
        u_mat = zeros(FT, NU, N)
        for i in 1:D; x_mat[i, :] .= x_flat[i][:]; end
        for i in 1:NU; u_mat[i, :] .= u_flat[i][:]; end

        # 4. Clean NaNs
        x_clean, u_clean = SFH.remove_nans(x_mat, u_mat)

        # 5. Delegate
        return SFC.calculate_structure_function(x_clean, u_clean, bin_edges, sf_type; kwargs...)
    end
end

# Aliases for .hdf5, .hdf
SFC.calculate_structure_function_from_file(::Val{:hdf5}, args...; kwargs...) = 
    SFC.calculate_structure_function_from_file(Val(:h5), args...; kwargs...)
SFC.calculate_structure_function_from_file(::Val{:hdf}, args...; kwargs...) = 
    SFC.calculate_structure_function_from_file(Val(:h5), args...; kwargs...)

function _load_h5vars(file, key)
    if key isa Union{Tuple, AbstractVector}
        return ntuple(i -> read(file[string(key[i])]), length(key))
    else
        val = read(file[string(key)])
        if val isa Union{Tuple, AbstractVector} && !(eltype(val) <: Number)
             return ntuple(i -> val[i], length(val))
        end
        return (val,)
    end
end

end # module
