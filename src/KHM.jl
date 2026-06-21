"""
Basic homogeneous/isotropic KHM exact-law diagnostics.

This module intentionally works on already-binned scalar outputs. Full KHMH or
AGKE budgets need additional fields such as viscosity, forcing, pressure,
inhomogeneous transport terms, and temporal tendencies.
"""
module KHM

using ..HelperFunctions: HelperFunctions as SFH

export bin_midpoints,
    finite_difference,
    transverse_incompressibility_residual,
    epsilon_from_four_fifths,
    four_fifths_residual

"""
    bin_midpoints(edges)

Midpoints of flat bin edges.
"""
bin_midpoints(edges) = SFH.midpoints(edges)

"""
    finite_difference(r, y)

Second-order centered finite difference on nonuniform coordinates, using
one-sided differences at the endpoints.
"""
function finite_difference(r::AbstractVector, y::AbstractVector)
    length(r) == length(y) ||
        throw(DimensionMismatch("r and y must have the same length"))
    n = length(r)
    n >= 2 || throw(ArgumentError("finite_difference requires at least two samples"))
    out = similar(y, float(eltype(y)))
    @inbounds begin
        out[1] = (y[2] - y[1]) / (r[2] - r[1])
        for i in 2:(n - 1)
            out[i] = (y[i + 1] - y[i - 1]) / (r[i + 1] - r[i - 1])
        end
        out[n] = (y[n] - y[n - 1]) / (r[n] - r[n - 1])
    end
    return out
end

"""
    transverse_incompressibility_residual(r, D_LL, D_TT; dimension=3)

Residual of the isotropic incompressibility relation
``D_TT = D_LL + r/(D-1) dD_LL/dr``.
"""
function transverse_incompressibility_residual(
    r::AbstractVector,
    D_LL::AbstractVector,
    D_TT::AbstractVector;
    dimension::Integer = 3,
)
    length(D_LL) == length(D_TT) == length(r) ||
        throw(DimensionMismatch("r, D_LL, and D_TT must have the same length"))
    dimension >= 2 || throw(ArgumentError("dimension must be >= 2"))
    dDdr = finite_difference(r, D_LL)
    return D_TT .- D_LL .- r .* dDdr ./ (dimension - 1)
end

"""
    epsilon_from_four_fifths(r, S3)

Pointwise dissipation estimate from the isotropic 3D four-fifths law
``S3(r) = -(4/5) ε r``.
"""
epsilon_from_four_fifths(r, S3) = .-(5 / 4) .* S3 ./ r

"""
    four_fifths_residual(r, S3, epsilon)

Residual of ``S3(r) = -(4/5) ε r``.
"""
four_fifths_residual(r, S3, epsilon) = S3 .+ (4 / 5) .* epsilon .* r

end
