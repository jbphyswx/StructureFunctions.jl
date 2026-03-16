module SyntheticData
using Random: Random
using StaticArrays: SVector
using LinearAlgebra: dot

export generate_nonuniform_domain, generate_spectral_field, generate_highres_uniform

"""
    generate_nonuniform_domain(N_total; lat_range=(-30.0, 30.0), lon_range=(0.0, 60.0), R_mask=0.0)

Generates a non-uniform domain by mapping a regular lat-lon grid to Cartesian (x, y).
Optionally applies a circular mask at the domain edge to simulate land.
"""
function generate_nonuniform_domain(N_side::Int; 
    lat_range=(-30.0, 30.0), 
    lon_range=(0.0, 60.0), 
    R_mask=0.0,
    x_land=nothing,
    y_land=nothing
)
    # 1. Create uniform lat-lon grid
    lats_grid = range(lat_range[1], stop=lat_range[2], length=N_side)
    lons_grid = range(lon_range[1], stop=lon_range[2], length=N_side)
    
    lats = Float64[]
    lons = Float64[]
    for lat in lats_grid, lon in lons_grid
        push!(lats, lat)
        push!(lons, lon)
    end
    
    # 2. Map to Cartesian (Mercator-ish or simple projection to induce non-uniformity)
    # x = R * cos(lat) * lon
    # y = R * lat
    # We'll use a simplified version:
    xs = lons .* cos.(deg2rad.(lats))
    ys = lats
    
    # 3. Apply analytic mask (circle at the edge)
    if R_mask > 0
        x_c = x_land === nothing ? maximum(xs) : x_land
        y_c = y_land === nothing ? minimum(ys) : y_land
        mask = .!((xs .- x_c).^2 .+ (ys .- y_c).^2 .< R_mask^2)
        return xs[mask], ys[mask], mask
    end
    
    return xs, ys, trues(length(xs))
end

"""
    generate_spectral_field(x, y; peaks=[], noise_level=0.1)

Generates a field u(x, y) with specified spectral peaks and noise.
peaks: List of (kx, ky, amplitude) tuples.
"""
function generate_spectral_field(xs, ys; peaks=[], noise_level=0.0)
    N = length(xs)
    FT = eltype(xs)
    u = zeros(Complex{FT}, N)
    
    # Add deterministic peaks
    for (kx, ky, amp) in peaks
        u .+= amp .* exp.(im .* (kx .* xs .+ ky .* ys))
    end
    
    # Add noise if requested
    if noise_level > 0
        u .+= noise_level .* (randn(FT, N) .+ im .* randn(FT, N))
    end
    
    return u
end

"""
    generate_highres_uniform(N_high; lat_range=(-30,30), lon_range=(0,60), peaks=[])

Generates a high-resolution uniform grid and analytic signal for sampling validation.
Useful for checking if NUFFT of scattered samples matches FFT of perfect high-res grid.
"""
function generate_highres_uniform(N_high::Int; 
    lat_range=(-30.0, 30.0), 
    lon_range=(0.0, 60.0), 
    peaks=[]
)
    lats_grid = range(lat_range[1], stop=lat_range[2], length=N_high)
    lons_grid = range(lon_range[1], stop=lon_range[2], length=N_high)
    
    # Simple Cartesian mapping
    xs = [lon * cos(deg2rad(lat)) for lat in lats_grid, lon in lons_grid]
    ys = [lat for lat in lats_grid, lon in lons_grid]
    
    u = zeros(ComplexF64, N_high, N_high)
    for (kx, ky, amp) in peaks
        u .+= amp .* exp.(im .* (kx .* xs .+ ky .* ys))
    end
    
    return xs, ys, u
end

end # module
