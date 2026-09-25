"""
Basic homogeneous/isotropic KHM exact-law diagnostics.

This module intentionally works on already-binned scalar outputs. Full KHMH or
AGKE budgets need additional fields such as viscosity, forcing, pressure,
inhomogeneous transport terms, and temporal tendencies.
"""
module KHM

export finite_difference,
    transverse_incompressibility_residual,
    epsilon_from_four_fifths,
    epsilon_from_four_thirds,
    four_thirds_residual,
    epsilon_theta_from_yaglom,
    yaglom_residual,
    four_fifths_residual

"""
    finite_difference(r, y)

Differentiate samples `y` on finite, strictly increasing coordinates `r`.

For at least three samples, use the quadratic interpolant through three adjacent
points, including one-sided endpoint formulas. These formulas are second-order
accurate on nonuniform grids. Two samples produce the secant slope at both points.
The output element type promotes the floating-point types of `r` and `y`.
"""
function finite_difference(r::AbstractVector, y::AbstractVector)
    length(r) == length(y) || throw(DimensionMismatch("r and y must have the same length"))
    Base.require_one_based_indexing(r, y)
    n = length(r)
    n >= 2 || throw(ArgumentError("finite_difference requires at least two samples"))
    eltype(r) <: Real || throw(ArgumentError("coordinates must be real"))
    all(isfinite, r) || throw(ArgumentError("coordinates must be finite"))
    all(i -> r[i] > r[i - 1], 2:n) || throw(ArgumentError("coordinates must be strictly increasing"))
    T = promote_type(float(eltype(r)), float(eltype(y)))
    out = similar(y, T)
    if n == 2
        fill!(out, (T(y[2]) - T(y[1])) / (T(r[2]) - T(r[1])))
        return out
    end
    @inbounds for i in 2:(n - 1)
        h1 = T(r[i]) - T(r[i - 1])
        h2 = T(r[i + 1]) - T(r[i])
        s1 = (T(y[i]) - T(y[i - 1])) / h1
        s2 = (T(y[i + 1]) - T(y[i])) / h2
        out[i] = s1 + h1 / (h1 + h2) * (s2 - s1)
        i == 2 && (out[1] = s1 - h1 / (h1 + h2) * (s2 - s1))
        i == n - 1 && (out[n] = s2 + h2 / (h1 + h2) * (s2 - s1))
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
    epsilon_from_four_fifths(r, L3)

Dissipation from Kolmogorov's four-fifths law, ``⟨δu_L³⟩ = -(4/5) ε r``.

`L3` is **`L3SF`**, the third-order *longitudinal* structure function ``⟨δu_L³⟩`` — not `S3SF`, which
is ``⟨δu_L‖δu‖²⟩`` and obeys the **four-thirds** law instead (see
[`epsilon_from_four_thirds`](@ref)). The two differ by a factor of 5/3, so passing the wrong one
returns a wrong `ε` that looks entirely reasonable.
"""
epsilon_from_four_fifths(r, L3) = .-(5 / 4) .* L3 ./ r

"""
    four_fifths_residual(r, L3, epsilon)

Residual of ``⟨δu_L³⟩ = -(4/5) ε r``. `L3` is `L3SF`, as in [`epsilon_from_four_fifths`](@ref).
"""
four_fifths_residual(r, L3, epsilon) = L3 .+ (4 / 5) .* epsilon .* r

"""
    epsilon_from_four_thirds(r, S3)

Dissipation from the four-thirds law, ``⟨δu_L‖δu‖²⟩ = -(4/3) ε r``.

`S3` is **`S3SF`**, the package's third-order *mixed* structure function. This is the law that
quantity obeys; `L3SF` obeys the four-fifths law (see [`epsilon_from_four_fifths`](@ref)).
"""
epsilon_from_four_thirds(r, S3) = .-(3 / 4) .* S3 ./ r

"""
    four_thirds_residual(r, S3, epsilon)

Residual of ``⟨δu_L‖δu‖²⟩ = -(4/3) ε r``.
"""
four_thirds_residual(r, S3, epsilon) = S3 .+ (4 / 3) .* epsilon .* r

"""
    epsilon_theta_from_yaglom(r, LS2)

Scalar dissipation from Yaglom's law, ``⟨δu_L (δθ)²⟩ = -(4/3) ε_θ r``.

`LS2` is the mixed velocity–scalar moment `MixedSFType{1,0,2}` computes. The quantity returned is
`ε_θ`, the dissipation of scalar variance — a different quantity from the `ε` of the velocity laws,
and not interchangeable with it.
"""
epsilon_theta_from_yaglom(r, LS2) = .-(3 / 4) .* LS2 ./ r

"""
    yaglom_residual(r, LS2, epsilon_theta)

Residual of ``⟨δu_L (δθ)²⟩ = -(4/3) ε_θ r``.
"""
yaglom_residual(r, LS2, epsilon_theta) = LS2 .+ (4 / 3) .* epsilon_theta .* r

end
