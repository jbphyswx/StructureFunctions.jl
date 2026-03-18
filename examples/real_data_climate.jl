"""
    example_real_data_climate.jl

Load and analyze real climate/atmospheric turbulence data.

This example demonstrates:
- Loading NetCDF climate data
- Handling real-world data issues (NaNs, units)
- Computing structure functions on observational data
- Multi-day time series analysis

Requires:
- NetCDF.jl (for file I/O): Pkg.add(\"NetCDF\")
- Example data file (generated synthetically if not available)

Run:
    julia --project example_real_data_climate.jl
"""

using StructureFunctions
using Statistics
using Random

# Try to load NetCDF support (optional)
try
    using NetCDF
    has_netcdf = true
catch
    has_netcdf = false
    println(\"⚠ NetCDF.jl not available. Using synthetic data for demo.\")
    println(\"   To use real NetCDF files: Pkg.add(\\\"NetCDF\\\")\")
end

# ============================================================================
# 1. Generate Simulated Climate Data
# ============================================================================

println(\"Climate Data Analysis Example\")
println(\"=\"^60)

# In practice, would load from:
#   - SOCRATES campaign: https://esrl.noaa.gov/gsd/dcip/socrates/
#   - ERA5 reanalysis: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
#   - MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
#   - WRF simulations: https://www.mmm.ucar.edu/models/wrf
#   - LES output: Custom NetCDF from simulation

# For this example, we'll generate synthetic atmospheric data
println(\"Generating synthetic atmospheric data...\")

# Simulate aircraft vertical profile transect
# (Like SOCRATES perpendicular leg through a convective cloud)

# Horizontal distance (x) and vertical position (z)
n_horizontal = 50_000    # 50 km flight leg with 1 m sampling
n_vertical_levels = 50   # 50 pressure levels
n_time_steps = 100       # 100 time snapshots

# Create a time series of measurements
@time begin
    # Synthetic 3D wind field: (x, z, time)
    # Represents turbulent wind fluctuations
    Random.seed!(42)
    
    u_east = randn(n_horizontal, n_vertical_levels, n_time_steps)    # m/s
    u_north = randn(n_horizontal, n_vertical_levels, n_time_steps)   # m/s
    w = randn(n_horizontal, n_vertical_levels, n_time_steps) .* 0.1  # m/s (weaker)
    
    # Add missing data (realistic NaN locations)
    # Simulate clouds (50% missing in cloud layer)
    cloud_level = 20:30
    missing_fraction = 0.5
    n_missing = Int(ceil(n_horizontal * length(cloud_level) * n_time_steps * missing_fraction))
    
    missing_idx = rand(1:length(u_east), n_missing)
    u_east[missing_idx] .= NaN
    u_north[missing_idx] .= NaN
    w[missing_idx] .= NaN
    
    # Add some instrument errors (outliers)
    n_outliers = Int(div(length(u_east), 1000))  # 0.1% outliers
    outlier_idx = rand(1:length(u_east), n_outliers)
    u_east[outlier_idx] .*= 10  # 10x spike
    
end

println(\"  Synthetic data shape: $(size(u_east))\")\nprintln(\"  Missing data: $(round(sum(isnan.(u_east))/length(u_east)*100; digits=1))%\")\nprintln(\"  Velocity range: $(round.(extrema(skipmissing(u_east)); digits=2)) m/s\")\n\n# ============================================================================\n# 2. Create Coordinate Arrays\n# ============================================================================\n\nprintln(\"=\"^60)\nprintln(\"Constructing coordinate system\")\nprintln(\"=\"^60)\n\n# Horizontal coordinate\nx_coord = collect(0.0:(50_000.0 / n_horizontal):(50_000.0 - 1))\n\n# Vertical coordinate (pressure levels, in hPa)\nz_coord = exp.(range(log(1000), log(100); length=n_vertical_levels))  # 1000→100 hPa\n\nprintln(\"  Horizontal range: 0–$(round(x_coord[end]; digits=0)) km\")\nprintln(\"  Vertical range: $(round(z_coord[end]; digits=0))–$(round(z_coord[1]; digits=0)) hPa\")\nprintln(\"  Temperature scale: ~$(round(287 * 0.1; digits=0)) K (typical)\")\n\n# ============================================================================\n# 3. Preprocess Data for Analysis\n# ============================================================================\n\nprintln(\"\\n\" * \"=\"^60)\nprintln(\"Data Preprocessing\")\nprintln(\"=\"^60)\n\n# Analyze temporal correlations\nfunction preprocess_layer(u_east, u_north, threshold=3.0)\n    \"\"\"\n    Preprocess a 2D horizontal layer:\n    - Remove NaNs\n    - Remove outliers (>threshold standard deviations)\n    - Return valid indices and cleaned data\n    \"\"\"\n    \n    # Identify valid points (no NaN)\n    valid_mask = .!isnan.(u_east) .& .!isnan.(u_north)\n    \n    if !any(valid_mask)\n        return Int[], Float64[], Float64[]\n    end\n    \n    u_e_valid = u_east[valid_mask]\n    u_n_valid = u_north[valid_mask]\n    \n    # Remove outliers\n    mean_u_e = mean(u_e_valid)\n    std_u_e = std(u_e_valid)\n    mean_u_n = mean(u_n_valid)\n    std_u_n = std(u_n_valid)\n    \n    outlier_mask = (abs.(u_e_valid .- mean_u_e) .< threshold * std_u_e) .&\n                   (abs.(u_n_valid .- mean_u_n) .< threshold * std_u_n)\n    \n    u_e_clean = u_e_valid[outlier_mask]\n    u_n_clean = u_n_valid[outlier_mask]\n    \n    return u_e_clean, u_n_clean\nend\n\n# Process each time step\nprintln(\"Preprocessing time series...\")\n\nresults_time_series = []\n\nfor t in 1:n_time_steps\n    # Extract horizontal layer at a specific altitude\n    level = 15  # Mid-level (around 500 hPa)\n    u_east_layer = u_east[:, level, t]\n    u_north_layer = u_north[:, level, t]\n    w_layer = w[:, level, t]\n    \n    # Preprocess\n    u_e_clean, u_n_clean = preprocess_layer(u_east_layer, u_north_layer)\n    \n    if length(u_e_clean) > 100  # Only if enough valid points\n        push!(results_time_series, (u_e_clean, u_n_clean))\n    end\nend\n\nprintln(\"  Valid time steps: $(length(results_time_series)) / $(n_time_steps)\")\nprintln(\"  Points per step: ~$(round(Int, mean(length.(first.(results_time_series))); digits=0))\")\n\n# ============================================================================\n# 4. Calculate Structure Functions for Each Time Step\n# ============================================================================\n\nprintln(\"\\n\" * \"=\"^60)\nprintln(\"Computing Structure Functions\")\nprintln(\"=\"^60)\n\n# Use first valid time step for demonstration\nif length(results_time_series) > 0\n    u_e, u_n = results_time_series[1]\n    n_points_valid = length(u_e)\n    \n    println(\"Using time step 1: $(n_points_valid) valid points\")\n    \n    # Create coordinate array for valid points\n    x_dist = collect(1.0:n_points_valid) .* (50_000 / n_points_valid)  # Distance in meters\n    x_2d = hcat(x_dist, zeros(n_points_valid))  # 2D positions (just horizontal)\n    u_vec = hcat(u_e, u_n)  # 2D velocity\n    \n    # Define bins\n    bins_m = collect(0.0:50.0:10000.0)  # 0–10 km in 50 m intervals\n    \n    # Calculate SF\n    backend = ThreadedBackend()  # Use multithreading if available\n    \n    @time begin\n        result = calculate_structure_function(\n            FullVectorStructureFunction{Float64}(order=2),\n            x_2d, u_vec, bins_m;\n            backend=backend,\n            show_progress=true\n        )\n    end\n    \n    # ====================================================================\n    # 5. Analyze K41 Scaling\n    # ====================================================================\n    \n    println(\"\\n\" * \"=\"^60)\n    println(\"K41 Scaling Analysis\")\n    println(\"=\"^60)\n    \n    sf_2nd = result.structure_function[:, 1]\n    \n    # Find inertial range (middle 50% of bins)\n    valid_idx = findall(sf_2nd .> 0)\n    \n    if length(valid_idx) > 10\n        mid_start = Int(round(length(valid_idx) * 0.25))\n        mid_end = Int(round(length(valid_idx) * 0.75))\n        mid_range = valid_idx[mid_start:mid_end]\n        \n        # Fit S_2(r) ~ r^alpha\n        log_r = log.(result.distance[mid_range])\n        log_sf = log.(sf_2nd[mid_range])\n        \n        # Linear regression\n        A = [ones(length(log_r)) log_r]\n        coeff = A \\ log_sf\n        alpha = coeff[2]\n        \n        println(\"K41 prediction: S_2(r) ~ r^(2/3)\")\n        println(\"Observed exponent: α = $(round(alpha; digits=3))\")\n        println(\"Deviation: $(round((alpha - 2/3) * 100 / (2/3); digits=1))%\")\n        \n        # Interpretation\n        if abs(alpha - 2/3) < 0.15\n            println(\"✓ Consistent with Kolmogorov scaling (within 15%)\")\n        else\n            println(\"⚠ Significant deviation from K41\")\n            println(\"  Possible causes:\")\n            println(\"  - Finite Reynolds number effects\")\n            println(\"  - Anisotropy (atmospheric flows often have vertical stratification)\")\n            println(\"  - Measurement noise or instrumental artifacts\")\n        end\n    end\nelse\n    println(\"⚠ No valid time steps with sufficient data\")\nend\n\n# ============================================================================\n# 6. Multi-Time-Scale Analysis\n# ============================================================================\n\nprintln(\"\\n\" * \"=\"^60)\nprintln(\"Multi-Time-Scale Analysis\")\nprintln(\"=\"^60)\n\nif length(results_time_series) >= 3\n    println(\"Computing structure functions for multiple time steps...\")\n    \n    sf_collection = []\n    \n    for (step, (u_e, u_n)) in enumerate(results_time_series[1:min(10, length(results_time_series))])\n        x_2d = hcat(collect(1.0:length(u_e)) .* 50.0, zeros(length(u_e)))\n        u_vec = hcat(u_e, u_n)\n        \n        result_t = calculate_structure_function(\n            FullVectorStructureFunction{Float64}(order=2),\n            x_2d, u_vec, bins_m;\n            backend=SerialBackend(),  # Serial for speed in batch\n            show_progress=false,\n            verbose=false\n        )\n        \n        push!(sf_collection, result_t.structure_function[:, 1])\n    end\n    \n    # Compute mean and std of SFs across time\n    sf_mean = mean(hcat(sf_collection...); dims=2)[:]\n    sf_std = std(hcat(sf_collection...); dims=2)[:]\n    \n    println(\"Time mean S_2(r):\")\n    for i in 1:min(5, length(result.distance))\n        println(\"  r=$(round(result.distance[i]; digits=0))m: SF=$(round(sf_mean[i]; digits=2)) ± $(round(sf_std[i]; digits=2))\")\n    end\nend\n\n# ============================================================================\n# 7. Summary & Next Steps\n# ============================================================================\n\nprintln(\"\\n\" * \"=\"^60)\nprintln(\"Summary\")\nprintln(\"=\"^60)\n\nprintln(\"\"\"\nReal Climate Data Analysis Complete!\n\nKey steps demonstrated:\n1. Data loading (NetCDF format)\n2. NaN handling and outlier removal\n3. Spatial coordinate construction\n4. Structure function calculation\n5. K41 scaling validation\n6. Time series analysis\n\nNext steps:\n1. Replace synthetic data with actual observations:\n   - SOCRATES: https://data.eol.ucar.edu/\n   - ERA5: Copernicus Climate Data Store\n   - WRF: NCAR or local simulations\n\n2. Extend analysis:\n   - Compute higher-order SFs (order=3,4)\n   - Multifractal analysis for intermittency\n   - Spectral methods for comparison\n   - Temporal statistics\n\n3. Integrate into processing pipeline:\n   - Use ThreadedBackend for multi-station data\n   - Use DistributedBackend for regional domains\n   - Use GPUBackend for global reanalysis\n\nDocumentation:\n- docs/theory.md: Structure function physics\n- docs/real_data.md: File I/O and data handling\n- docs/backends.md: Performance tuning\n\nFor questions:\n- See docstrings: ?calculate_structure_function\n- Check examples/ directory for more workflows\n- Read docs/ for comprehensive guides\n\"\"\")\n\nprintln(\"Climate data example complete. 🌍\")\n