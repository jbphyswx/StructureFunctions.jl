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
    if W === nothing
        scales = SFC._relative_scales(y)
        Kw, yw = K ./ scales, y ./ scales
    else
        Kw, yw = SFC._whitened(K, y, W)
    end
    model(_, p) = Kw * SFC.segmented_spectrum(p, k, edges)
    lo, hi = m.slope_bounds
    lower = vcat(0.0, fill(lo, S))
    upper = vcat(Inf, fill(hi, S))
    α0 = clamp(-5 / 3, lo, hi)
    unit = model(r, vcat(1.0, fill(α0, S)))
    b0 = max(sum(abs, yw) / max(sum(abs, unit), eps()), eps())
    p0 = vcat(b0, fill(α0, S))
    fit = LsqFit.curve_fit(model, r, Vector{Float64}(yw), LsqFit.PrecisionWeights(ones(length(yw))), p0; lower, upper)
    return fit.param, Matrix(LsqFit.vcov(fit)), edges, fit.converged
end

end # module
