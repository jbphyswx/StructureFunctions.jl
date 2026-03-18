module StructureFunctionsJLD2Ext

using JLD2: JLD2
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH

"""
    calculate_structure_function_from_file(::Val{:jld2}, fpath::String, bin_edges, sf_type; 
                                          x_key="x", u_key="u", kwargs...)

Load data from a JLD2 file and calculate the structure function.

# Arguments
- `fpath`: Path to the .jld2 file.
- `x_key`: Key for position data (can be a String or Tuple of Strings for multiple dims).
- `u_key`: Key for velocity data (can be a String or Tuple of Strings for multiple components).
"""
function SFC.calculate_structure_function_from_file(
    ::Val{:jld2},
    fpath::String,
    bin_edges,
    sf_type;
    x_key = "x",
    u_key = "u",
    kwargs...,
)
    JLD2.jldopen(fpath, "r") do file
        # 1. Load data
        x_raw = _load_keys(file, x_key)
        u_raw = _load_keys(file, u_key)

        # 2. Flatten if multi-dim (Block I3)
        x_flat = SFH.flatten_data(x_raw)
        u_flat = SFH.flatten_data(u_raw)

        # 3. Convert to matrices (D, N) and (NU, N)
        FT = eltype(x_flat[1])
        D = length(x_flat)
        NU = length(u_flat)
        N = length(x_flat[1])

        x_mat = zeros(FT, D, N)
        u_mat = zeros(FT, NU, N)
        for i in 1:D
            x_mat[i, :] .= x_flat[i]
        end
        for i in 1:NU
            u_mat[i, :] .= u_flat[i]
        end

        # 4. Clean NaNs (Block I3)
        x_clean, u_clean = SFH.remove_nans(x_mat, u_mat)

        # 5. Delegate to core calculation
        return SFC.calculate_structure_function(
            sf_type,
            x_clean,
            u_clean,
            bin_edges;
            kwargs...,
        )
    end
end

# Helper to load multiple keys if provided as Tuple/Vector
function _load_keys(file, key)
    if key isa Union{Tuple, AbstractVector}
        return ntuple(i -> file[string(key[i])], length(key))
    else
        val = file[string(key)]
        # If the value itself is a Tuple/Vector of data, return it
        if val isa Union{Tuple, AbstractVector} && !(eltype(val) <: Number)
            return ntuple(i -> val[i], length(val))
        end
        return (val,)
    end
end

end # module
