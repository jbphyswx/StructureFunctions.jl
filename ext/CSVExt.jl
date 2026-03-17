module CSVExt

using CSV
import StructureFunctions.Calculations: calculate_structure_function_from_file, calculate_structure_function
import StructureFunctions.HelperFunctions: remove_nans

"""
    calculate_structure_function_from_file(::Val{:csv}, fpath::String, bin_edges, sf_type; 
                                          x_key=("x1", "x2"), u_key=("u1", "u2"), kwargs...)

Load data from a CSV file and calculate the structure function.
"""
function calculate_structure_function_from_file(
    ::Val{:csv},
    fpath::String,
    bin_edges,
    sf_type;
    x_key = ("x1", "x2"),
    u_key = ("u1", "u2"),
    kwargs...
)
    # Load CSV file (using threads if available)
    df = CSV.File(fpath)
    
    # 1. Load columns
    x_keys = x_key isa Union{Tuple, AbstractVector} ? x_key : (x_key,)
    u_keys = u_key isa Union{Tuple, AbstractVector} ? u_key : (u_key,)
    
    D = length(x_keys)
    NU = length(u_keys)
    N = length(df)
    
    # Use first row to get eltype
    FT = Float64
    try
        FT = typeof(getproperty(df[1], Symbol(u_keys[1])))
    catch; end

    x_mat = zeros(FT, D, N)
    u_mat = zeros(FT, NU, N)

    for (i, k) in enumerate(x_keys)
        x_mat[i, :] .= Array(getproperty(df, Symbol(k)))
    end
    for (i, k) in enumerate(u_keys)
        u_mat[i, :] .= Array(getproperty(df, Symbol(k)))
    end

    # 2. Clean NaNs
    x_clean, u_clean = remove_nans(x_mat, u_mat)

    # 3. Delegate
    return calculate_structure_function(x_clean, u_clean, bin_edges, sf_type; kwargs...)
end

# Alias for .txt, .dat
calculate_structure_function_from_file(::Val{:txt}, args...; kwargs...) = 
    calculate_structure_function_from_file(Val(:csv), args...; kwargs...)
calculate_structure_function_from_file(::Val{:dat}, args...; kwargs...) = 
    calculate_structure_function_from_file(Val(:csv), args...; kwargs...)

end # module
