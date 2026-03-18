"""
    simple_2d.jl

Basic 2D structure function calculation with strict qualified imports.

Run from package root:
    julia --project=examples examples/simple_2d.jl
"""

using StructureFunctions: StructureFunctions as SF
using Statistics: Statistics
using CairoMakie: CairoMakie as CM

println("Generating synthetic 2D turbulent data...")

grid_size = 128
spacing = 10.0
N = grid_size^2

x_1d = collect(0:spacing:((grid_size - 1) * spacing))
y_1d = collect(0:spacing:((grid_size - 1) * spacing))

xx = repeat(x_1d; inner = grid_size)
yy = repeat(y_1d; outer = grid_size)
x = hcat(xx, yy)

u_raw = randn(N, 2)
v_raw = randn(N, 2)
scale = sqrt.(1 .+ (1:N) ./ 1000)
u = hcat(u_raw[:, 1] ./ scale, v_raw[:, 1] ./ scale)

println("  Domain: $(grid_size) x $(grid_size) grid")
println("  Spacing: $(spacing) m")
println("  Total points: $(N)")
println("  Mean velocity: $(round.(Statistics.mean(u; dims = 1); digits = 2)) m/s")
println("  Velocity std: $(round.(Statistics.std(u; dims = 1); digits = 2)) m/s")

operator = SF.FullVectorStructureFunctionType{Float64}(order = 2)
backend = SF.SerialBackend()

r_min = 10.0
r_max = grid_size * spacing / 2
n_bins = 30
bins = exp.(range(log(r_min), log(r_max); length = n_bins))

result = @time SF.calculate_structure_function(
    operator,
    x,
    u,
    bins;
    backend = backend,
    show_progress = true,
    verbose = true,
)

sf_2 = result.structure_function[:, 1]
valid_idx = sf_2 .> 0
mid_idx = nothing

if any(valid_idx)
    r_valid = result.distance[valid_idx]
    sf_valid = sf_2[valid_idx]
    i0 = max(1, Int(floor(length(r_valid) * 0.25)))
    i1 = max(i0, Int(ceil(length(r_valid) * 0.75)))
    mid_idx = i0:i1

    log_r = log.(r_valid[mid_idx])
    log_sf = log.(sf_valid[mid_idx])
    A = hcat(ones(length(log_r)), log_r)
    coeff = A \ log_sf
    slope = coeff[2]

    println("  K41 prediction slope: 0.667")
    println("  Observed slope: $(round(slope; digits = 3))")
end

fig = CM.Figure(size = (760, 520))
ax = CM.Axis(
    fig[1, 1],
    xlabel = "Separation [m]",
    ylabel = "S2(r)",
    xscale = CM.log10,
    yscale = CM.log10,
    title = "2nd-Order Structure Function",
)
CM.scatterlines!(ax, result.distance, sf_2, label = "S2", markersize = 6)

if mid_idx !== nothing
    mid_r = result.distance[mid_idx]
    k41_line = 10.0 .* mid_r .^ (2 / 3)
    CM.lines!(ax, mid_r, k41_line, label = "K41 r^(2/3)", linestyle = :dash)
end

CM.axislegend(ax, position = :lt)
CM.save("sf_simple_2d_example.png", fig)

println("Saved plot to sf_simple_2d_example.png")
