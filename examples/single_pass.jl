"""
    single_pass.jl

Single-pass structure functions: all six isotropic invariants (S2, L2, T2, S3, L3, L1T2) plus
the Helmholtz rotational/divergent decomposition, computed in ONE O(N²) pair pass.

Run from the package root:
    julia --project=examples examples/single_pass.jl
"""

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using Random: Random
using CairoMakie: CairoMakie as CM

Random.seed!(0)

# Synthetic 2D point cloud (D, N) with a decaying-increment structure.
N = 4096
x = rand(2, N) .* 1.0e4
u = randn(2, N)

# Log-spaced distance bins as a fast O(1) LogBinEdges wrapper.
edges = collect(exp10.(range(log10(50.0), log10(5.0e3); length = 41)))
bins = SF.LogBinEdges(edges)

println("Single pass over N=$N points, $(length(edges) - 1) distance bins...")

# Averaged invariants (default output_type = StructureFunction). One pass returns a NamedTuple
# keyed by invariant; point-field input also yields a `:helmholtz` entry.
res = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = CB.AutoBackend())

println("Invariants returned: ", keys(res))
@assert haskey(res, :helmholtz) "point-field input should include the Helmholtz decomposition"

# Raw sums + counts instead of the averaged view:
raw = SFC.calculate_structure_functions_single_pass(
    x, u, bins; output_type = SF.StructureFunctionSumsAndCounts,
)
println("S2 total pair count: ", sum(raw.S2.counts))

# Plot the averaged invariants + rotational/divergent on log-log axes. `.distance` holds the bin
# EDGES (length n_bins + 1); use midpoints for the per-bin values (length n_bins).
rmid = 0.5 .* (edges[1:(end - 1)] .+ edges[2:end])
fig = CM.Figure(; size = (820, 520))
ax = CM.Axis(
    fig[1, 1];
    xscale = log10, yscale = log10, xlabel = "separation r", ylabel = "|S(r)|",
    title = "Single-pass invariants (N=$N)",
)
for k in (:S2, :L2, :T2, :S3, :L3, :L1T2)
    v = getproperty(res, k).values
    CM.lines!(ax, rmid, abs.(v) .+ 1e-12; label = string(k))
end
# Rotational/divergent are stored as raw sums + counts → average them per bin.
h = res.helmholtz
rot = h.rotational_sums ./ max.(h.rotational_counts, 1)
div = h.divergent_sums ./ max.(h.divergent_counts, 1)
CM.lines!(ax, rmid, abs.(rot) .+ 1e-12; linestyle = :dash, label = "Rotational")
CM.lines!(ax, rmid, abs.(div) .+ 1e-12; linestyle = :dash, label = "Divergent")
CM.axislegend(ax; position = :rb, nbanks = 2)

out = joinpath(@__DIR__, "single_pass.png")
CM.save(out, fig)
println("Saved figure to ", out)
