module StructureFunctionsNCDatasetsExt

using NCDatasets: NCDatasets as NC
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH

"""
    calculate_structure_function_from_file(::Val{:nc}, fpath::String, bin_edges, sf_type; 
                                          x_key=("lon", "lat"), u_key=("u", "v"), kwargs...)

Load data from a NetCDF file and calculate the structure function.
"""
function SFC.calculate_structure_function_from_file(
    ::Val{:nc},
    fpath::String,
    bin_edges,
    sf_type;
    x_key = ("lon", "lat"),
    u_key = ("u", "v"),
    kwargs...,
)
    NC.Dataset(fpath, "r") do ds
        # 1. Load data variables
        x_raw = _load_vars(ds, x_key)
        u_raw = _load_vars(ds, u_key)

        # 2. Get values and flatten
        x_vals = ntuple(i -> x_raw[i][:], length(x_raw))
        u_vals = ntuple(i -> u_raw[i][:], length(u_raw))

        x_flat = SFH.flatten_data(x_vals)
        u_flat = SFH.flatten_data(u_vals)

        # 3. Handle coordinate grids (using lazy u_raw[1] for shape)
        x_flat_expanded = _expand_coords(x_flat, u_raw[1])

        # 4. Convert to matrices
        FT = Float64
        try
            FT = eltype(u_flat[1])
        catch
        end

        D = length(x_flat_expanded)
        NU = length(u_flat)
        N = length(u_flat[1])

        x_mat = zeros(FT, D, N)
        u_mat = zeros(FT, NU, N)
        for i in 1:D
            x_mat[i, :] .= x_flat_expanded[i]
        end
        for i in 1:NU
            u_mat[i, :] .= u_flat[i]
        end

        # 5. Clean NaNs (common in NetCDF)
        x_clean, u_clean = SFH.remove_nans(x_mat, u_mat)

        # 6. Delegate
        return SFC.calculate_structure_function(
            sf_type,
            x_clean,
            u_clean,
            bin_edges;
            kwargs...,
        )
    end
end

# Alias for .netcdf
SFC.calculate_structure_function_from_file(::Val{:netcdf}, args...; kwargs...) =
    SFC.calculate_structure_function_from_file(Val(:nc), args...; kwargs...)

function _load_vars(ds, key)
    if key isa Union{Tuple, AbstractVector}
        return ntuple(i -> ds[string(key[i])], length(key))
    else
        return (ds[string(key)],)
    end
end

function _expand_coords(x_flat, u_sample)
    # If x_flat[1] length matches u_sample length, they are already expanded or match.
    N_u = length(u_sample)
    if length(x_flat[1]) == N_u
        return x_flat
    end

    # Otherwise, assume x_flat contains 1D coordinate vectors that need to be gridded.
    # e.g. x_flat = (lon, lat, z), u_sample = (lon_sz, lat_sz, z_sz)
    sz = size(u_sample)
    if length(x_flat) == length(sz)
        # Expand using broadcasting
        # lon along dim 1, lat along dim 2, etc.
        expanded = ntuple(
            d -> begin
                # Create a shape with 1s everywhere except dimension d
                new_sz = ntuple(i -> i == d ? length(x_flat[d]) : 1, length(sz))
                field = reshape(x_flat[d], new_sz)
                # Broadcast to full size
                full_field = zeros(eltype(field), sz...)
                full_field .= field
                return vec(full_field)
            end, length(x_flat))
        return expanded
    end

    return x_flat # Fallback
end

end # module
