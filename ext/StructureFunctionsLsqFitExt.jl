module StructureFunctionsLsqFitExt

using LsqFit: LsqFit
using LinearAlgebra: LinearAlgebra as LA
using StructureFunctions: Calculations as SFC

# The bounded Levenberg–Marquardt fit of a segmented power law: the kernel matrix over the quadrature
# nodes is fixed, so the model is that matrix times the spectrum at the nodes and the Jacobian comes
# from forward-mode differentiation of the spectrum alone.
function SFC._segmented_fit(
    m::SFC.SegmentedPowerLaw, ::Val{D}, r::AbstractVector, y::AbstractVector, W, k_lo, k_hi,
) where {D}
    S = m.segments
    edges = SFC._segment_edges(k_lo, k_hi, S)
    k, _, K = SFC._segmented_design(Val(D), r, edges)
    model(_, p) = K * SFC.segmented_spectrum(p, k, edges)
    lo, hi = m.slope_bounds
    lower = vcat(0.0, fill(lo, S))
    upper = vcat(Inf, fill(hi, S))
    α0 = clamp(-5 / 3, lo, hi)
    unit = model(r, vcat(1.0, fill(α0, S)))
    b0 = max(sum(abs, y) / max(sum(abs, unit), eps()), eps())
    p0 = vcat(b0, fill(α0, S))
    fit = LsqFit.curve_fit(model, r, Vector{Float64}(y), _weights(W, y), p0; lower, upper)
    return fit.param, Matrix(LsqFit.vcov(fit)), edges, fit.converged
end

# No data covariance fits the relative residual (S₂ᶠⁱᵗ − S₂)/S₂; variances weigh each value by its
# own; a full covariance whitens by its inverse.
_weights(::Nothing, y) = LsqFit.PrecisionWeights(1 ./ abs2.(Vector{Float64}(y)))
_weights(W::AbstractVector, y) = LsqFit.PrecisionWeights(1 ./ Vector{Float64}(W))
_weights(W::AbstractMatrix, y) = LsqFit.PrecisionMatrix(inv(LA.Symmetric(Matrix{Float64}(W))))

end # module
