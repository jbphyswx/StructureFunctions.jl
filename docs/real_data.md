# Real Data: File I/O & Preprocessing

This document explains how to work with real climate and ocean data, including NaN handling, file formats, and preprocessing strategies.

## Table of Contents
- [File Format Overview](#file-format-overview)
- [NaN Handling](#nan-handling)
- [Data Preprocessing](#data-preprocessing)
- [Format-Specific Guides](#format-specific-guides)
- [Common Workflows](#common-workflows)
- [Performance Tips](#performance-tips)

---

## File Format Overview

| Format | Use Case | Status | Speed | Memory |
|--------|----------|--------|-------|--------|
| **NetCDF** | Climate models, observational data | ✅ Supported | Medium | Medium |
| **JLD2** | Julia-native binary (fastest save/load) | ✅ Supported | Fast | Low |
| **HDF5** | General scientific data | ✅ Supported | Medium | Medium |
| **CSV** | Manual inspection, spreadsheets | ✅ Supported | Slow | High |
| **Zarr** | Cloud storage (S3, GCS) | 🟡 Planned v0.4 | Medium | Medium |
| **Binary (raw)** | High performance, custom formats | ✅ Supported | Fast | Very Low |
| **GRIB** | Weather/climate (advanced) | ⚠️ No support yet | Slow | High |

### Recommended Choice

- **For development**: CSV (human-readable, easy to inspect)
- **For computation**: NetCDF or JLD2 (balance of speed and compatibility)
- **For production**: JLD2 (fastest Julia I/O) or Zarr (cloud-native)
- **For archival**: NetCDF (widely supported by community)

---

## NaN Handling

### Why NaNs Occur

Real data contains missing/invalid values:
- Sensor failures or gaps
- Land pixels in ocean data
- Clouds in remote sensing
- Quality control filters

### Default Behavior

**StructureFunctions.jl by default skips any point pair with NaN**:

```julia
# If ANY component of u_i or u_j is NaN,
# the entire pair is skipped silently

result = calculate_structure_function(x, u, bins)

# Check how many pairs were used:
@show sum(result.counts)  # Total valid pairs
```

### Strategies for NaN Data

#### Strategy 1: Filter Before Calculation (Recommended)

```julia
using Statistics

# Remove rows with ANY NaN
valid_idx = vec(all(!isnan.(u); dims=2)) .& vec(all(!isnan.(x); dims=2))
x_clean = x[valid_idx, :]
u_clean = u[valid_idx, :]

result = calculate_structure_function(x_clean, u_clean, bins)
```

**Pros**:
- ✅ Clear what was removed
- ✅ Faster (less computation on invalid pairs)
- ✅ Simpler debugging

**Cons**:
- ✗ May lose structured patterns (e.g., entire rows of ocean data)

#### Strategy 2: Imputation by Interpolation

```julia
# Interpolate missing velocity over valid neighbors
# (Only do this if justified by physics!)

using Interpolations

# Locate NaNs
nan_mask = isnan.(u)

# Build interpolator from valid data
valid_idx = .!nan_mask
itp = LinearInterpolation(x[valid_idx, :], u[valid_idx, :])

# Impute (CAUTION: introduces artificial correlations)
u_interp = copy(u)
u_interp[nan_mask] .= itp(x[nan_mask, :])

result = calculate_structure_function(x, u_interp, bins)
```

**Pros**:
- ✅ Preserves dataset size
- ✅ Uses spatial structure

**Cons**:
- ✗ Introduces BIAS (artificial smoothness)
- ✗ Violates independence assumption
- ✗ Only valid for sparse, localized NaNs

**Recommendation**: Only interpolate if <5% data is missing and spatially clustered.

#### Strategy 3: Stratified Calculation

```julia
# Calculate SFs separately for different regions/times
# (E.g., land vs ocean, day vs night)

ocean_mask = is_ocean_pixel.(x)
land_data = (x[.!ocean_mask], u[.!ocean_mask])
ocean_data = (x[ocean_mask], u[ocean_mask])

# Separate calculations
result_land = calculate_structure_function(land_data..., bins)
result_ocean = calculate_structure_function(ocean_data..., bins)

# Combine results (with caution!)
combined = merge_structure_functions(result_land, result_ocean)
```

**Pros**:
- ✅ Respects physical heterogeneity
- ✅ Reveals domain-specific properties

**Cons**:
- ✗ More complex bookkeeping
- ✗ Reduced sample size per region

---

## Data Preprocessing

### Unit Conversion

**Ensure consistent units before calculation**:

```julia
# Example: Temperature fluctuations in Kelvin
T = read_netcdf("temperature.nc")  # K

# Convert velocity from m/s to same scale as temperature
# (If T is in K and x is in meters, velocities should be compatible)
u_raw = read_netcdf("wind_speed.nc")  # m/s

# Option A: Normalize both
u = (u_raw .- mean(u_raw)) ./ std(u_raw)
T = (T .- mean(T)) ./ std(T)

# Option B: Convert to physical units
x = [i * 10e-3 for i in 1:size(u, 1)]  # 10 mm spacing
u = u_raw  # Keep in m/s

result = calculate_structure_function(x, u, bins)
```

### Detrending

**Remove low-frequency trends**:

```julia
using FFTW

# Remove mean
u_demean = u .- vec(mean(u; dims=1))'

# Remove linear trend
for col in 1:size(u, 2)
    u[:, col] .-= detrend_linear(u[:, col])
end

# Remove mean and standard normalization
u_std = (u_demean) ./ vec(std(u_demean; dims=1))'

result = calculate_structure_function(x, u_std, bins)
```

### Coordinate Systems

**Match coordinates to domain**:

```julia
# Example 1: Regular grid
lon = -180:0.1:180  # Longitude
lat = -90:0.1:90    # Latitude
u = randn(length(lat), length(lon))  # Wind data

# Construct coordinate array
x = [(lo, la) for la in lat, lo in lon]  # 2D positions
result = calculate_structure_function(x[:], u[:], bins)

# Example 2: Irregular measurements (e.g., aircraft track)
t = 0:0.01:10  # Time (seconds)
lon = [...]; lat = [...]  # Gps coordinates
u = [...]  # Velocity measurements

x = hcat(lon, lat, t)  # 3D positions: (lon, lat, time)
result = calculate_structure_function(x, u, bins)
```

---

## Format-Specific Guides

### NetCDF (Climate Models, Observations)

NetCDF is the standard format for climate and oceanographic data.

#### Reading NetCDF

```julia
using NetCDF, StructureFunctions

# Open file
ds = open_netcdf("output.nc")

# Inspect variables
@show keys(ds)  # List variable names
@show size(ds["u"])  # Shape of variable

# Extract data
u_raw = ds["u"][:, :, :]  # Load entire array (or use subsetting below)
v_raw = ds["v"][:, :, :]
x_raw = ds["x"][:]
y_raw = ds["y"][:]

close(ds)

# Convert to format needed
x = hcat(x_raw, y_raw)  # Concatenate x, y coordinates
u = sqrt.(u_raw.^2 .+ v_raw.^2)  # Magnitude of wind

result = calculate_structure_function(x, u, 0:10:1000)
```

#### Subsetting Large Files

For very large NetCDF (terabytes), subset before loading:

```julia
using NetCDF

# Read only specific time/space
ds = open_netcdf("huge_output.nc")

# Subset: time 100:200, lat 20:80, lon 30:130
u_subset = ds["u"][100:200, 20:80, 30:130]

close(ds)

result = calculate_structure_function(x, u_subset, bins)
```

### JLD2 (Julia-Native Binary)

Fastest for saving/loading Julia-specific data structures.

#### Writing JLD2

```julia
using StructureFunctions, JLD2

# Calculate
result = calculate_structure_function(...)

# Save entire result object
save_object("sf_result.jld2", result)

# Or save individual arrays
jldopen("sf_result.jld2", "w") do file
    file["sf"] = result.structure_function
    file["distances"] = result.distance
    file["metadata"] = Dict("date" => now(), "version" => "0.3.0")
end
```

#### Reading JLD2

```julia
using JLD2

# Load entire result
result = load_object("sf_result.jld2")

# Or load individual arrays
data = load("sf_result.jld2")
sf_values = data["sf"]
distances = data["distances"]
metadata = data["metadata"]
```

**Performance**: 10–100x faster than NetCDF for Julia objects.

### CSV (Manual Inspection)

CSV is human-readable but slow for large files.

#### Writing CSV

```julia
using CSV, DataFrames

# Create DataFrame
df = DataFrame(
    x = vec(x[:, 1]),
    y = vec(x[:, 2]),
    u = vec(u[:, 1]),
    v = vec(u[:, 2])
)

# Write
CSV.write("data.csv", df)
```

#### Reading CSV

```julia
using CSV, DataFrames

# Read
df = CSV.read("data.csv", DataFrame)

# Extract columns
x = Matrix(df[:, [:x, :y]])
u = Matrix(df[:, [:u, :v]])

result = calculate_structure_function(x, u, bins)
```

**Caution**: Slow for >100M rows; use only for exploration.

### HDF5 (General Scientific Data)

Similar to NetCDF but more flexible; often used for custom formats.

#### Reading HDF5

```julia
using HDF5

# Open file
f = h5open("data.h5", "r")

# Inspect structure
@show keys(f)  # Top-level groups

# Read data
u = read(f, "velocity")
x = read(f, "coordinates")

close(f)

result = calculate_structure_function(x, u, bins)
```

---

## Common Workflows

### Workflow 1: SOCRATES Campaign Data

Processing atmospheric observations from SOCRATES campaign (aircraft):

```julia
using StructureFunctions, NetCDF, Dates

# Load data
file = "SOCRATES_leg1.nc"
ds = open_netcdf(file)

# Extract trajectory
lon = ds["lon"][:]
lat = ds["lat"][:]
alt = ds["alt"][:]
time = ds["time"][:]  # seconds since reference

# Velocity from IRS (inertial reference system)
u_north = ds["u_wind"][:]
u_east = ds["v_wind"][:]

# Position: geographic + altitude
x = hcat(lon, lat, alt)

# Velocity: horizontal wind
u = hcat(u_north, u_east)

close(ds)

# Preprocess: remove NaNs
valid = vec(.!any(isnan.([x, u]); dims=2))
x_valid = x[valid, :]
u_valid = u[valid, :]

# Calculate SF with threaded backend
backend = ThreadedBackend()
bins = 0:1:50  # km separation

result = calculate_structure_function(
    FullVectorStructureFunction{Float64}(order=2),
    x_valid, u_valid, bins;
    backend=backend,
    show_progress=true
)

# Analyze multifractal properties
@show result.structure_function
```

### Workflow 2: Climate Model Validation

Comparing a climate model to observations:

```julia
using StructureFunctions, NetCDF

# Load model simulation
model_file = "model_output.nc"
ds_model = open_netcdf(model_file)
u_model = ds_model["u"][:, :, 1:1000]  # Subset for memory
x_model = construct_grid_coordinates(ds_model)
close(ds_model)

# Load observations
obs_file = "observations.nc"
ds_obs = open_netcdf(obs_file)
u_obs = ds_obs["u"][:, :, 1:1000]
x_obs = construct_grid_coordinates(ds_obs)
close(ds_obs)

# Calculate both
backend = AutoBackend()  # Use available resources
bins = 0:10:500

result_model = calculate_structure_function(x_model, u_model, bins; backend)
result_obs = calculate_structure_function(x_obs, u_obs, bins; backend)

# Compare
using Plots
plot(result_model.distance, result_model.structure_function[:, 1],
     label="Model", title="2nd-Order Structure Function")
plot!(result_obs.distance, result_obs.structure_function[:, 1],
      label="Observations")
```

### Workflow 3: Zarr Cloud Data (Future)

Reading data from cloud storage (requires v0.4+):

```julia
# Planned for v0.4
using StructureFunctions, Zarr

# Open cloud-hosted Zarr
url = "s3://climate-data-bucket/ERA5/2023.zarr"
ds = open_zarr(url)

# Streaming subset
data = ds["wind"][1:1000, 1:1000, :]  # Only load what you need

x_cloud = construct_coordinates(ds)
u_cloud = extract_velocity(data, ds.metadata)

# Distributed calculation
backend = DistributedBackend()
result = calculate_structure_function(x_cloud, u_cloud, bins; backend)
```

---

## Performance Tips

### Tip 1: Profile Before Optimization

```julia
using Profile, ProfileCanvas

# Identify bottleneck
@profile result = calculate_structure_function(...)

# Generate flame graph
ProfileCanvas.show()

# Common bottlenecks:
# - I/O (file reading)
# - Preprocessing (NaN removal, unit conversion)
# - Calculation (dispatch to backend)
# - Post-processing (result normalization)
```

### Tip 2: Use Float32 for Large Data

```julia
# Memory: Float32 = half the memory of Float64
# Precision: OK for most turbulence analysis
# Speed: 2x faster on GPU

x = Float32.(rand(1_000_000_000, 2))
u = Float32.(randn(1_000_000_000, 2))

backend = GPUBackend()
result = calculate_structure_function(
    FullVectorStructureFunction{Float32}(order=2),
    x, u, bins; backend
)
```

### Tip 3: Batch Processing

```julia
# Process multiple time slices incrementally
# (Instead of loading all at once)

results = []
for t in 1:100
    # Load one time step
    u_t = load_time_slice(file, t)
    x_t = construct_coordinates(...)
    
    # Calculate
    result_t = calculate_structure_function(x_t, u_t, bins)
    push!(results, result_t)
end

# Combine results
combined = merge_structure_functions(results...)
```

### Tip 4: Lazy Evaluation

```julia
# Don't convert units until needed
# Don't detrend until needed
# Keep in original format as long as possible

x_raw = mmap("data.bin")  # Memory-mapped file
u_raw = mmap("velocity.bin")

# Pass directly to calculation
# (Backend will convert if needed)
result = calculate_structure_function(x_raw, u_raw, bins; backend=AutoBackend())
```

---

## Related Topics

- [Backends](backends.md): Performance tuning for your data size
- [Architecture](architecture.md): Extension mechanism for custom formats
- [Theory](theory.md): What structure functions measure
- [Examples](../examples/README.md): Complete workflows with real data
