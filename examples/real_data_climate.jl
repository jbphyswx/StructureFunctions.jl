"""
    real_data_climate.jl

Real-data style workflow with strict qualified imports.

Run from package root:
    julia --project=examples examples/real_data_climate.jl
"""

using StructureFunctions: StructureFunctions as SF
using Statistics: Statistics
using Random: Random

Random.seed!(42)

n_horizontal = 20_000
n_vertical = 20
n_time = 8

u_east = randn(n_horizontal, n_vertical, n_time)
u_north = randn(n_horizontal, n_vertical, n_time)

missing_idx = rand(1:length(u_east), Int(floor(0.1 * length(u_east))))
u_east[missing_idx] .= NaN
u_north[missing_idx] .= NaN

function preprocess_layer(u_east_layer, u_north_layer)
    valid_mask = .!isnan.(u_east_layer) .& .!isnan.(u_north_layer)
    return u_east_layer[valid_mask], u_north_layer[valid_mask]
end

u_e, u_n = preprocess_layer(u_east[:, 10, 1], u_north[:, 10, 1])

x_dist = collect(1.0:length(u_e)) .* 50.0
x = hcat(x_dist, zeros(length(u_e)))
u = hcat(u_e, u_n)

operator = SF.FullVectorStructureFunctionType{Float64}(order = 2)
bins = collect(0.0:100.0:5000.0)

result = @time SF.calculate_structure_function(
    operator,
    x,
    u,
    bins;
    backend = SF.ThreadedBackend(),
    show_progress = true,
    verbose = false,
)

sf_vals = result.structure_function[:, 1]
mean_sf = Statistics.mean(sf_vals)
std_sf = Statistics.std(sf_vals)

println("Valid points: $(length(u_e))")
println("S2 mean: $(round(mean_sf; digits = 4))")
println("S2 std:  $(round(std_sf; digits = 4))")
