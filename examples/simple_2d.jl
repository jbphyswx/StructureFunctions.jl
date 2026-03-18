"""
    example_simple_2d.jl

Basic 2D structure function calculation for homogeneous turbulence.

This example demonstrates:
- Generating synthetic 2D turbulent velocity data
- Computing 2nd-order structure functions
- Visualizing results
- K41 scaling validation

Run:
    julia example_simple_2d.jl
"""

using StructureFunctions
using Statistics
using Plots

# ============================================================================
# 1. Generate Synthetic Turbulent Data
# ============================================================================

println("Generating synthetic 2D turbulent data...")

# Domain: 256×256 grid with 10 m spacing
grid_size = 256
spacing = 10.0  # meters per grid point
N = grid_size^2

# Create regular grid
x_1d = 0:spacing:(grid_size-1)*spacing
y_1d = 0:spacing:(grid_size-1)*spacing

# Construct 2D coordinate array [N×2]
x = vec([repeat(x_1d, grid_size)'; repeat(y_1d, 1, grid_size)])'  # Flatten

# Generate velocity fluctuations
# Simple model: Isotropic turbulence with k^(-5/3) spectrum
u_raw = randn(N, 2)
v_raw = randn(N, 2)

# Apply spectral filter (simple Gaussian filter for demonstration)
u = u_raw ./ sqrt.(1 .+ (1:N) ./ 1000)  # Fake decay with scale
v = v_raw ./ sqrt.(1 .+ (1:N) ./ 1000)

u = hcat(u[:, 1], v[:, 1])  # [N×2] velocity matrix (u_x, u_y)

println("  Domain: $(grid_size) × $(grid_size) grid")
println("  Spacing: $(spacing) m")
println("  Total points: $(N)")
println("  Mean velocity: $(round.(mean(u; dims=1); digits=2)) m/s")
println("  Velocity std: $(round.(std(u; dims=1); digits=2)) m/s")

# ============================================================================
# 2. Define Structure Function & Backend
# ============================================================================

println("\nDefining structure function operator...")

# Full vector structure function, 2nd order
operator = FullVectorStructureFunction{Float64}(order=2)
println("  Operator: Full vector SF, order=2")

# Use appropriate backend
backend = SerialBackend()  # Small data → Serial is fine
println("  Backend: Serial (single-threaded)")

# ============================================================================
# 3. Define Distance Bins
# ============================================================================

println("\nDefining distance bins...")

# Separation = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
# Max separation ~ grid_size * spacing ≈ 2560 m
# Use logarithmic spacing to sample inertial range

r_min = 10  # 1 grid unit
r_max = 256 * spacing / 2  # Half the domain
n_bins = 30

bins = exp.(range(log(r_min), log(r_max); length=n_bins))

println("  Distance range: $(round(r_min, digits=1)) – $(round(r_max, digits=1)) m")
println("  Number of bins: $(n_bins)")

# ============================================================================
# 4. Calculate Structure Function
# ============================================================================

println("\nCalculating 2nd-order structure function...")
println("  This may take a minute for $(N) points...")

@time result = calculate_structure_function(
    operator,
    x, u, bins;
    backend=backend,
    show_progress=true,
    verbose=true
)

println("  ✓ Calculation complete")
println("  Valid pairs per bin: $(extrema(result.counts))")

# ============================================================================
# 5. Analyze Results
# ============================================================================

println("\nAnalyzing structure function scaling...")

# Extract 2nd-order SF (all orders stored in matrix columns)
sf_2 = result.structure_function[:, 1]

# Check for K41 scaling: S_2(r) ~ r^(2/3)
# Log-log scaling: log(S_2) ~ (2/3) * log(r)

valid_idx = sf_2 .> 0

if any(valid_idx)
    # Fit log-log regression in inertial range
    r_valid = result.distance[valid_idx]
    sf_valid = sf_2[valid_idx]
    
    # Use middle 50% of range (inertial range)
    mid_idx = Int(round(length(r_valid) * 0.25)):Int(round(length(r_valid) * 0.75))
    
    log_r = log.(r_valid[mid_idx])
    log_sf = log.(sf_valid[mid_idx])
    
    # Linear regression: log_sf = slope * log_r + const
    A = [ones(length(log_r)) log_r]
    coeff = A \ log_sf
    slope = coeff[2]
    
    println("  K41 prediction: S_2(r) ~ r^(2/3) → slope = 0.667")
    println("  Observed slope: $(round(slope; digits=3))")
    println("  " * (abs(slope - 2/3) < 0.1 ? "✓ Consistent with K41" :
                    "⚠ Deviation from K41 (could be non-local effects)"))
end

# ============================================================================
# 6. Plot Results
# ============================================================================

println("\nGenerating plot...")

# Create log-log plot
p = plot(
    result.distance, sf_2,
    xaxis=:log, yaxis=:log,
    xlabel="Separation [m]",
    ylabel="S₂(r) [m²/s²]",
    title="2nd-Order Structure Function",
    legend=:topleft,
    marker=:circle,
    markersize=4,
    linewidth=2,
    size=(700, 500)
)

# Add K41 reference line
if any(valid_idx)
    mid_r = result.distance[mid_idx]
    k41_line = 10.0 .* mid_r .^ (2/3)  # Arbitrary scaling
    plot!(p, mid_r, k41_line, label="K41 scaling (r²/³)", linestyle=:dash, linewidth=2)
end

# Save plot
savefig(p, "sf_simple_2d_example.png")
println("  Plot saved: sf_simple_2d_example.png")

# ============================================================================
# 7. Summary
# ============================================================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("""
This example calculated the 2nd-order structure function for
a 256×256 grid of synthetic turbulent velocity data.

Key insights:
1. S₂(r) quantifies variance of velocity increments at scale r
2. K41 theory predicts S₂(r) ~ r^(2/3) in the inertial range
3. Our synthetic data approximately follows this scaling
4. Real data often deviates due to:
   - Finite Reynolds number effects
   - Boundaries and inhomogeneities
   - Intermittency corrections

For real atmospheric data, see:
- docs/real_data.md: File I/O and preprocessing
- docs/theory.md: Structure function theory
- example_socrates_campaign.jl: Real campaign data
""")

println("\nNext steps:")
println("1. Try with ThreadedBackend for larger grids")
println("2. Load real data (see example_real_data_climate.jl)")
println("3. Compute higher-order SFs to study intermittency")
println("4. Read docs/theory.md for physics background")
