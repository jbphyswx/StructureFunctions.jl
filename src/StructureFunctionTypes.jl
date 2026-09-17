module StructureFunctionTypes

using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA
using ..HelperFunctions: HelperFunctions as SFH
using ..MultiFields: MultiFields as MF

"""
    AbstractStructureFunctionType

The operator of a structure function: which statistic of a pair's increment is accumulated.
"""
abstract type AbstractStructureFunctionType end

"""
    AbstractPairwiseStructureFunctionType

An operator with a value on one pair, `sf(δu, r̂)`, that the pair loops, the gridded sweeps and the
transforms accumulate.
"""
abstract type AbstractPairwiseStructureFunctionType <: AbstractStructureFunctionType end

"""
    AbstractDerivedStructureFunctionType

A quantity derived from binned structure functions, with no value on one pair.
"""
abstract type AbstractDerivedStructureFunctionType <: AbstractStructureFunctionType end

# Identity call: allows `SFType()` for singleton operator instances.
(sf::AbstractStructureFunctionType)() = sf

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

"""
    norm2(x)

Compute the sum of squares of elements of `x`. Faster than `norm(x)^2`
for small vectors, used for transverse components.
"""
@inline function norm2(x)
    @fastmath @inbounds begin
        out = zero(eltype(x))
        for i in eachindex(x)
            out += x[i]^2
        end
        return out
    end
end

# ---------------------------------------------------------------------------
# Parametric Types
# ---------------------------------------------------------------------------

"""
    ProjectedStructureFunctionType{NL, NT}(basis = CanonicalTransverseBasis())

Operator `δu_L^NL · δu_T^NT`: `NL` powers of the longitudinal increment `δu·r̂` and `NT` of the
transverse one. `NT = 2` is the invariant transverse energy `‖δu‖² − δu_L²`; every other `NT` is the
signed component along the first vector of `transverse_basis(basis, r̂)`, so the sign of an odd `NT`
is the convention's. [`SFH.CanonicalTransverseBasis`](@ref) is `n̂`, the turn about `ẑ`. The basis travels
with the operator into every result.

The two readings differ only for `D ≥ 3`, where the transverse plane has more than one direction; in
2-D `‖δu_T‖² = (δu·n̂)²`. They cannot be unified: `‖δu_T‖^NT` is a polynomial in `δu` only for even
`NT`, since `‖δu_T‖² = ‖δu‖² − δu_L²` and an odd power needs the square root, so odd `NT` admits the
component reading alone. `NT = 2` is the energy because that is the quantity the second-order
relations are written in — the Helmholtz split and the `D_LL`/`D_TT` isotropy relations all take it.
[`TransverseComponentSecondOrderStructureFunctionType`](@ref) is the per-component form,
`‖δu_T‖²/(D−1)`.
"""
struct ProjectedStructureFunctionType{NL, NT, B <: SFH.AbstractTransverseBasisConvention} <:
       AbstractPairwiseStructureFunctionType
    basis::B
end

"""Alias of [`ProjectedStructureFunctionType`](@ref)."""
const ProjectedStructureFunction = ProjectedStructureFunctionType

ProjectedStructureFunctionType{NL, NT}(
    basis::B = SFH.CanonicalTransverseBasis(),
) where {NL, NT, B <: SFH.AbstractTransverseBasisConvention} = ProjectedStructureFunctionType{NL, NT, B}(basis)

ProjectedStructureFunctionType(NL::Integer, NT::Integer, basis = SFH.CanonicalTransverseBasis()) =
    ProjectedStructureFunctionType{NL, NT}(basis)

"""
    (sf::ProjectedStructureFunctionType{NL,NT})(δu, r̂)

Compute the structure function kernel for longitudinal/transverse components.

- `NL` : power of longitudinal component δu_l
- `NT` : power of transverse component ||δu_t||
"""
@generated function (sf::ProjectedStructureFunctionType{NL, NT})(δu_in, r̂) where {NL, NT}
    ex = :(one(eltype(δu)))

    # Longitudinal contribution (always scalar, integer power)
    if !iszero(NL)
        if NL == 1
            ex = :($ex * SFH.mδu_l(δu, r̂))
        elseif NL == 2
            ex = :($ex * SFH.mδu_l(δu, r̂)^2)
        else
            ex = :($ex * (SFH.mδu_l(δu, r̂)^$NL))
        end
    end

    # Transverse contribution: NT = 2 is the invariant energy; any other NT is the signed component
    # along the operator's basis.
    if !iszero(NT)
        if NT == 2
            ex = :($ex * SFH.transverse_norm2(δu, r̂))
        elseif NT == 1
            ex = :($ex * SFH.mδu_t(δu, r̂, sf.basis))
        else
            ex = :($ex * (SFH.mδu_t(δu, r̂, sf.basis)^$NT))
        end
    end

    return quote
        δu = SFC_field_vector(δu_in, 1)
        $ex
    end
end

# ---------------------------------------------------------------------------
"""
    SecondOrderStructureFunctionType()

Full-vector second-order structure function, ``S2SF = ||δu||²``.
"""
struct SecondOrderStructureFunctionType <: AbstractPairwiseStructureFunctionType end

"""
    ThirdOrderStructureFunctionType()

Third-order scalar flux structure function,
``S3SF = δu_L * ||δu||² = L3SF + L1T2SF``.
It is intentionally not ``||δu||³``.
"""
struct ThirdOrderStructureFunctionType <: AbstractPairwiseStructureFunctionType end

"""
    FullVectorStructureFunctionType{NF}

Generic full-vector norm-power operator, ``||δu||^NF``. `S3SF`'s conventional definition is
[`ThirdOrderStructureFunctionType`](@ref).
"""
struct FullVectorStructureFunctionType{NF} <: AbstractPairwiseStructureFunctionType end

"""
    MomentTensorOperator{P}()

The transform engine's request for the whole rank-`P` increment moment tensor
`M[i₁, …, i_P] = Σ_pairs Π_k δu[i_k]` of a field's vector field, in place of one scalar contraction
of it. It has no value on a single pair. In a fixed Cartesian frame an odd rank changes sign when a
pair is read from its other end, exactly as an odd scalar increment does, so it takes the same
canonical pair reading; in a pair's own geodesic frame the components are read the same from either
end and no reading enters.
"""
struct MomentTensorOperator{P} <: AbstractPairwiseStructureFunctionType end

(::MomentTensorOperator)(δu, r̂) = throw(ArgumentError(
    "the moment tensor operator has no value on one pair; it names the whole increment moment tensor for the transform engine",
))

"""Alias of [`FullVectorStructureFunctionType`](@ref)."""
const FullVectorStructureFunction = FullVectorStructureFunctionType

FullVectorStructureFunctionType(NF::Integer) = FullVectorStructureFunctionType{NF}()

@inline (::SecondOrderStructureFunctionType)(δu, r̂) = norm2(SFC_field_vector(δu, 1))

@inline function (::ThirdOrderStructureFunctionType)(δu, r̂)
    v = SFC_field_vector(δu, 1)
    return SFH.mδu_l(v, r̂) * norm2(v)
end

@generated function (::FullVectorStructureFunctionType{NF})(δu, r̂) where {NF}
    NF == 2 && return :(norm2(SFC_field_vector(δu, 1)))
    return :(LA.norm(SFC_field_vector(δu, 1))^$NF)
end

"""
    TransverseComponentSecondOrderStructureFunctionType()

Per-component transverse second-order structure function,
``||δu_t||² / (D - 1)``. This is distinct from `T2SF`, which stores the total
transverse energy.
"""
struct TransverseComponentSecondOrderStructureFunctionType <: AbstractPairwiseStructureFunctionType end

"""
    LongitudinalTransverseComponentThirdOrderStructureFunctionType()

Per-component variant of `L1T2SF`,
``δu_L * ||δu_t||² / (D - 1)``.
"""
struct LongitudinalTransverseComponentThirdOrderStructureFunctionType <: AbstractPairwiseStructureFunctionType end

@inline (::TransverseComponentSecondOrderStructureFunctionType)(δu, r̂) =
    SFH.transverse_component_norm2(SFC_field_vector(δu, 1), r̂)

@inline function (::LongitudinalTransverseComponentThirdOrderStructureFunctionType)(δu, r̂)
    v = SFC_field_vector(δu, 1)
    return SFH.mδu_l(v, r̂) * SFH.transverse_component_norm2(v, r̂)
end

# ---------------------------------------------------------------------------
# Named Constants: Type Aliases (longhand and shorthands)

"""`ProjectedStructureFunctionType{2, 0}`: the longitudinal second-order structure function ``⟨δu_L²⟩``."""
const LongitudinalSecondOrderStructureFunctionType = ProjectedStructureFunctionType{2, 0}
"""`ProjectedStructureFunctionType{0, 2}`: the transverse second-order structure function ``⟨‖δu_T‖²⟩ = ⟨‖δu‖² − δu_L²⟩``."""
const TransverseSecondOrderStructureFunctionType = ProjectedStructureFunctionType{0, 2}
"""Shorthand for [`TransverseComponentSecondOrderStructureFunctionType`](@ref)."""
const T2ComponentSFType = TransverseComponentSecondOrderStructureFunctionType

"""`ProjectedStructureFunctionType{3, 0}`: the longitudinal third-order structure function ``⟨δu_L³⟩``."""
const DiagonalConsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{3, 0}
"""`ProjectedStructureFunctionType{2, 1}`: ``⟨δu_L² δu_T⟩`` with the signed transverse component of the operator's basis."""
const DiagonalInconsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{2, 1}
"""`ProjectedStructureFunctionType{1, 2}`: ``⟨δu_L ‖δu_T‖²⟩``."""
const OffDiagonalInconsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{1, 2}
"""`ProjectedStructureFunctionType{0, 3}`: ``⟨δu_T³⟩`` with the signed transverse component of the operator's basis."""
const OffDiagonalConsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{0, 3}

"""Shorthand for [`SecondOrderStructureFunctionType`](@ref), ``⟨‖δu‖²⟩``."""
const S2SFType = SecondOrderStructureFunctionType
"""Shorthand for [`LongitudinalSecondOrderStructureFunctionType`](@ref), ``⟨δu_L²⟩``."""
const L2SFType = LongitudinalSecondOrderStructureFunctionType
"""Shorthand for [`TransverseSecondOrderStructureFunctionType`](@ref), ``⟨‖δu_T‖²⟩``."""
const T2SFType = TransverseSecondOrderStructureFunctionType
"""Shorthand for [`ThirdOrderStructureFunctionType`](@ref), ``⟨δu_L ‖δu‖²⟩``."""
const S3SFType = ThirdOrderStructureFunctionType
"""Shorthand for [`DiagonalConsistentThirdOrderStructureFunctionType`](@ref), ``⟨δu_L³⟩``."""
const L3SFType = DiagonalConsistentThirdOrderStructureFunctionType
"""Shorthand for [`OffDiagonalConsistentThirdOrderStructureFunctionType`](@ref), ``⟨δu_T³⟩``."""
const T3SFType = OffDiagonalConsistentThirdOrderStructureFunctionType
"""Shorthand for [`DiagonalInconsistentThirdOrderStructureFunctionType`](@ref), ``⟨δu_L² δu_T⟩``."""
const L2T1SFType = DiagonalInconsistentThirdOrderStructureFunctionType
"""Shorthand for [`OffDiagonalInconsistentThirdOrderStructureFunctionType`](@ref), ``⟨δu_L ‖δu_T‖²⟩``."""
const L1T2SFType = OffDiagonalInconsistentThirdOrderStructureFunctionType
"""Shorthand for [`LongitudinalTransverseComponentThirdOrderStructureFunctionType`](@ref)."""
const L1T2ComponentSFType = LongitudinalTransverseComponentThirdOrderStructureFunctionType

# ---------------------------------------------------------------------------
# Named Constants: Singleton Functors (The "Longhand" names now point to instances)

"""The instance `SecondOrderStructureFunctionType()`."""
const SecondOrderStructureFunction = SecondOrderStructureFunctionType()
"""The instance `LongitudinalSecondOrderStructureFunctionType()`."""
const LongitudinalSecondOrderStructureFunction =
    LongitudinalSecondOrderStructureFunctionType()
"""The instance `TransverseSecondOrderStructureFunctionType()`."""
const TransverseSecondOrderStructureFunction = TransverseSecondOrderStructureFunctionType()
"""The instance `TransverseComponentSecondOrderStructureFunctionType()`."""
const T2ComponentSF = TransverseComponentSecondOrderStructureFunctionType()

"""The instance `ThirdOrderStructureFunctionType()`."""
const ThirdOrderStructureFunction = ThirdOrderStructureFunctionType()
"""The instance `DiagonalConsistentThirdOrderStructureFunctionType()`."""
const DiagonalConsistentThirdOrderStructureFunction =
    DiagonalConsistentThirdOrderStructureFunctionType()
"""The instance `DiagonalInconsistentThirdOrderStructureFunctionType()`."""
const DiagonalInconsistentThirdOrderStructureFunction =
    DiagonalInconsistentThirdOrderStructureFunctionType()
"""The instance `OffDiagonalInconsistentThirdOrderStructureFunctionType()`."""
const OffDiagonalInconsistentThirdOrderStructureFunction =
    OffDiagonalInconsistentThirdOrderStructureFunctionType()
"""The instance `OffDiagonalConsistentThirdOrderStructureFunctionType()`."""
const OffDiagonalConsistentThirdOrderStructureFunction =
    OffDiagonalConsistentThirdOrderStructureFunctionType()
"""The instance `LongitudinalTransverseComponentThirdOrderStructureFunctionType()`."""
const L1T2ComponentSF = LongitudinalTransverseComponentThirdOrderStructureFunctionType()

"""The instance `S2SFType()`."""
const S2SF = SecondOrderStructureFunction
"""The instance `L2SFType()`."""
const L2SF = LongitudinalSecondOrderStructureFunction
"""The instance `T2SFType()`."""
const T2SF = TransverseSecondOrderStructureFunction
"""The instance `S3SFType()`."""
const S3SF = ThirdOrderStructureFunction
"""The instance `L3SFType()`."""
const L3SF = DiagonalConsistentThirdOrderStructureFunction
"""The instance `T3SFType()`."""
const T3SF = OffDiagonalConsistentThirdOrderStructureFunction
"""The instance `L2T1SFType()`."""
const L2T1SF = DiagonalInconsistentThirdOrderStructureFunction
"""The instance `L1T2SFType()`."""
const L1T2SF = OffDiagonalInconsistentThirdOrderStructureFunction

# ---------------------------------------------------------------------------
# Raw-geometry evaluation (pair kernels)
# ---------------------------------------------------------------------------

"""
    _sf_raw(sf, δu, dx, r2)

Evaluate `sf` from raw pair geometry: separation `dx`, its squared length `r2 = dx⋅dx`, and `δu`.

Identical to `sf(δu, dx/√r2)` for every operator. The second-order specializations below are
polynomials in `p = δu⋅dx` and `‖δu‖²` over a power of `r²`, so they need no `sqrt`; odd orders and
odd transverse orders need `r̂` and fall through to the generic method.
"""
@inline _sf_raw(sf::AbstractStructureFunctionType, δu, dx, r2) = sf(δu, dx / sqrt(r2))

@inline _sf_raw(::SecondOrderStructureFunctionType, δu, dx, r2) = norm2(δu)

@inline function _sf_raw(::ProjectedStructureFunctionType{2, 0}, δu, dx, r2)
    p = LA.dot(δu, dx)
    return p * p / r2
end

@inline function _sf_raw(::ProjectedStructureFunctionType{0, 2}, δu, dx, r2)
    p = LA.dot(δu, dx)
    return norm2(δu) - p * p / r2
end

@inline function _sf_raw(::TransverseComponentSecondOrderStructureFunctionType, δu, dx, r2)
    D = length(dx)
    D > 1 || throw(ArgumentError(
        "T2ComponentSF averages over the transverse directions, of which there are none at D = 1",
    ))
    p = LA.dot(δu, dx)
    return (norm2(δu) - p * p / r2) / (D - 1)
end

"""
    ScalarStructureFunctionType{P}(field = 1)

``⟨(δθ)^P⟩`` on scalar `field` — the scalar structure function. `P = 2` is the Obukhov–Corrsin
quantity; odd `P` measures the skewness of the tracer increment.
"""
struct ScalarStructureFunctionType{P} <: AbstractPairwiseStructureFunctionType
    field::Int
end

ScalarStructureFunctionType{P}() where {P} = ScalarStructureFunctionType{P}(1)
"""Shorthand for [`ScalarStructureFunctionType`](@ref)."""
const ScalarSFType = ScalarStructureFunctionType

@inline (sf::ScalarStructureFunctionType{P})(δu, r̂) where {P} =
    SFC_field_scalar(δu, sf.field)^P

"""
    MixedStructureFunctionType{NL, NT, P}(vector_field = 1, scalar_field = 1)

``⟨δu_L^{NL} ‖δu_T‖^{NT} (δθ)^P⟩`` — a velocity–scalar mixed moment.

`{1, 0, 2}` is Yaglom's law, ``⟨δu_L (δθ)²⟩ = −(4/3) ε_θ r``; `{1, 0, 1}` is the flux of the tracer
itself. The velocity part is read from a transported vector field and the scalar part from a
differenced scalar field, so the two never mix frames.
"""
struct MixedStructureFunctionType{NL, NT, P} <: AbstractPairwiseStructureFunctionType
    vector_field::Int
    scalar_field::Int
end

MixedStructureFunctionType{NL, NT, P}() where {NL, NT, P} =
    MixedStructureFunctionType{NL, NT, P}(1, 1)
"""Shorthand for [`MixedStructureFunctionType`](@ref)."""
const MixedSFType = MixedStructureFunctionType

@inline function (sf::MixedStructureFunctionType{NL, NT, P})(δu, r̂) where {NL, NT, P}
    v = SFC_field_vector(δu, sf.vector_field)
    θ = SFC_field_scalar(δu, sf.scalar_field)
    l = SFH.mδu_l(v, r̂)
    t2 = SFH.transverse_norm2(v, r̂)
    return l^NL * sqrt(t2)^NT * θ^P
end

"""
    ScalarDotStructureFunctionType(a, b)

``⟨δθ^{(a)} δθ^{(b)}⟩`` — a second-order **cross-field** scalar moment.

`(1, 1)` is the scalar structure function. `(a, b)` with `a ≠ b` is what an advective structure
function is: `⟨δω δ𝓐_ω⟩` is this with `ω` and its advection as the two fields.
"""
struct ScalarDotStructureFunctionType <: AbstractPairwiseStructureFunctionType
    a::Int
    b::Int
end

const ScalarDotSFType = ScalarDotStructureFunctionType

@inline (sf::ScalarDotStructureFunctionType)(δu, r̂) =
    SFC_field_scalar(δu, sf.a) * SFC_field_scalar(δu, sf.b)

"""
    VectorDotStructureFunctionType(a, b)

``⟨δu^{(a)} · δu^{(b)}⟩`` — a second-order **cross-field** vector moment.

`(1, 1)` **is** `S2SF`: the existing second-order operator is this one's diagonal, not a separate
thing. `(a, b)` with `a ≠ b` is `⟨δu · δ𝓐_u⟩`, the advective structure function, which holds without
isotropy — the reason it is worth having beside the third-order laws.
"""
struct VectorDotStructureFunctionType <: AbstractPairwiseStructureFunctionType
    a::Int
    b::Int
end

const VectorDotSFType = VectorDotStructureFunctionType

@inline (sf::VectorDotStructureFunctionType)(δu, r̂) =
    LA.dot(SFC_field_vector(δu, sf.a), SFC_field_vector(δu, sf.b))

# How an operator reaches a field of an increment. A single-field increment is the plain
# vector every existing operator takes, so naming field 1 of it is the vector itself — that is what
# keeps `Fields(vectors = (u,))` identical to a bare `u`.
@inline SFC_field_vector(δu::MF.FieldIncrement{D, 0, K}, i::Integer) where {D, K} =
    throw(ArgumentError(
        "this field carries no vector fields, so a velocity operator has nothing to read. Build " *
        "it with Fields(vectors = (...), ...), or use a scalar operator.",
    ))

@inline SFC_field_vector(δu::MF.FieldIncrement, i::Integer) = MF.vector_field(δu, i)
@inline function SFC_field_vector(δu, i::Integer)
    i == 1 || throw(ArgumentError(
        "this field has one vector field; asked for field $i. Build the field with " *
        "Fields(vectors = (...), ...) to carry more.",
    ))
    return δu
end

@inline SFC_field_scalar(δu::MF.FieldIncrement, i::Integer) = MF.scalar_field(δu, i)
@inline SFC_field_scalar(δu, i::Integer) = throw(ArgumentError(
    "this field carries no scalar fields; asked for scalar field $i. Build the field with " *
    "Fields(scalars = (...), ...) to carry one.",
))

"""
    RotationalSecondOrderStructureFunctionType()

Helmholtz-derived 2D rotational second-order component. A derived quantity, computed from binned
`L2SF`/`T2SF` with `helmholtz_decompose_2d`.
"""
struct RotationalSecondOrderStructureFunctionType <: AbstractDerivedStructureFunctionType end

"""
    DivergentSecondOrderStructureFunctionType()

Helmholtz-derived 2D divergent second-order component. A derived quantity, computed from binned
`L2SF`/`T2SF` with `helmholtz_decompose_2d`.
"""
struct DivergentSecondOrderStructureFunctionType <: AbstractDerivedStructureFunctionType end

"""
    HelmholtzDecomposition2DType()

Derived quantity describing the 2D isotropic Helmholtz decomposition into
rotational and divergent second-order components.
"""
struct HelmholtzDecomposition2DType <: AbstractDerivedStructureFunctionType end

"""The instance `RotationalSecondOrderStructureFunctionType()`."""
const RotationalSecondOrderStructureFunction = RotationalSecondOrderStructureFunctionType()
"""The instance `DivergentSecondOrderStructureFunctionType()`."""
const DivergentSecondOrderStructureFunction = DivergentSecondOrderStructureFunctionType()
"""The instance `HelmholtzDecomposition2DType()`."""
const HelmholtzDecomposition2DOperator = HelmholtzDecomposition2DType()

# ---------------------------------------------------------------------------
# Convenience Mappings

const SF_TYPE_MAP = Dict{Symbol, AbstractStructureFunctionType}(
    :SecondOrderStructureFunction => SecondOrderStructureFunction,
    :LongitudinalSecondOrderStructureFunction =>
        LongitudinalSecondOrderStructureFunction,
    :TransverseSecondOrderStructureFunction => TransverseSecondOrderStructureFunction,
    :T2ComponentSF => T2ComponentSF,
    :RotationalSecondOrderStructureFunction => RotationalSecondOrderStructureFunction,
    :DivergentSecondOrderStructureFunction => DivergentSecondOrderStructureFunction,
    :HelmholtzDecomposition2D => HelmholtzDecomposition2DOperator,
    :ThirdOrderStructureFunction => ThirdOrderStructureFunction,
    :DiagonalConsistentThirdOrderStructureFunction =>
        DiagonalConsistentThirdOrderStructureFunction,
    :DiagonalInconsistentThirdOrderStructureFunction =>
        DiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalInconsistentThirdOrderStructureFunction =>
        OffDiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalConsistentThirdOrderStructureFunction =>
        OffDiagonalConsistentThirdOrderStructureFunction,
    :L2SF => L2SF,
    :T2SF => T2SF,
    :RotationalSF => RotationalSecondOrderStructureFunction,
    :DivergentSF => DivergentSecondOrderStructureFunction,
    :L3SF => L3SF,
    :S2SF => S2SF,
    :S3SF => S3SF,
    :T3SF => T3SF,
    :L2T1SF => L2T1SF,
    :L1T2SF => L1T2SF,
    :L1T2ComponentSF => L1T2ComponentSF,
)

export AbstractStructureFunctionType,
    AbstractPairwiseStructureFunctionType,
    AbstractDerivedStructureFunctionType,
    LongitudinalSecondOrderStructureFunctionType,
    TransverseSecondOrderStructureFunctionType,
    SecondOrderStructureFunctionType,
    ThirdOrderStructureFunctionType,
    DiagonalConsistentThirdOrderStructureFunctionType,
    DiagonalInconsistentThirdOrderStructureFunctionType,
    OffDiagonalInconsistentThirdOrderStructureFunctionType,
    OffDiagonalConsistentThirdOrderStructureFunctionType,
    RotationalSecondOrderStructureFunctionType,
    DivergentSecondOrderStructureFunctionType,
    HelmholtzDecomposition2DType,
    TransverseComponentSecondOrderStructureFunctionType,
    LongitudinalTransverseComponentThirdOrderStructureFunctionType,
    S2SFType, L2SFType, T2SFType, S3SFType, L3SFType, T3SFType, L2T1SFType, L1T2SFType,
    T2ComponentSFType, L1T2ComponentSFType,
    SecondOrderStructureFunction,
    LongitudinalSecondOrderStructureFunction,
    TransverseSecondOrderStructureFunction,
    RotationalSecondOrderStructureFunction,
    DivergentSecondOrderStructureFunction,
    HelmholtzDecomposition2DOperator,
    ThirdOrderStructureFunction,
    DiagonalConsistentThirdOrderStructureFunction,
    DiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalConsistentThirdOrderStructureFunction,
    S2SF, L2SF, T2SF, S3SF, L3SF, T3SF, L2T1SF, L1T2SF,
    T2ComponentSF, L1T2ComponentSF, ProjectedStructureFunctionType,
    FullVectorStructureFunctionType, MomentTensorOperator,
    ScalarStructureFunctionType, MixedStructureFunctionType,
    ScalarDotStructureFunctionType, VectorDotStructureFunctionType,
    ScalarSFType, MixedSFType, ScalarDotSFType, VectorDotSFType,
    ProjectedStructureFunction,
    FullVectorStructureFunction,
    get_structure_function_type

get_structure_function_type(x::String) = get_structure_function_type(Symbol(x))
function get_structure_function_type(x::Symbol)
    if haskey(SF_TYPE_MAP, x)
        return SF_TYPE_MAP[x]
    else
        error("Unknown structure function type symbol: $x")
    end
end

@generated function get_structure_function_type(::Val{sym}) where {sym}
    if haskey(SF_TYPE_MAP, sym)
        return Meta.quot(SF_TYPE_MAP[sym])
    else
        return :(error("Unknown structure function type symbol: $($sym)"))
    end
end

"""
    get_structure_function_type(order::Int, mode::Symbol)

Map an integer order and a mode symbol (e.g., :longitudinal, :transverse, :scalar/total)
to a specific operator instance.
"""
function get_structure_function_type(order::Int, mode::Symbol)
    if order == 2
        if mode ∈ (:longitudinal, :long, :L)
            return LongitudinalSecondOrderStructureFunction
        elseif mode ∈ (:transverse, :trans, :T)
            return TransverseSecondOrderStructureFunction
        elseif mode ∈ (:transverse_component, :trans_component, :Tcomponent)
            return T2ComponentSF
        elseif mode ∈ (:scalar, :total, :S, :full)
            return SecondOrderStructureFunction
        elseif mode ∈ (:rotational, :rot)
            return RotationalSecondOrderStructureFunction
        elseif mode ∈ (:divergent, :div)
            return DivergentSecondOrderStructureFunction
        end
    elseif order == 3
        if mode ∈ (:longitudinal, :long, :L, :diagonal_consistent)
            return DiagonalConsistentThirdOrderStructureFunction
        elseif mode ∈ (:transverse, :trans, :T, :off_diagonal_consistent)
            return OffDiagonalConsistentThirdOrderStructureFunction
        elseif mode ∈ (:scalar, :total, :S, :full)
            return ThirdOrderStructureFunction
        elseif mode == :diagonal_inconsistent
            return DiagonalInconsistentThirdOrderStructureFunction
        elseif mode == :off_diagonal_inconsistent
            return OffDiagonalInconsistentThirdOrderStructureFunction
        elseif mode ∈ (:off_diagonal_inconsistent_component, :L1T2_component)
            return L1T2ComponentSF
        end
    end
    error("No mapping for order $order and mode $mode")
end

@generated function get_structure_function_type(
    ::Val{order},
    ::Val{mode},
) where {order, mode}
    return Meta.quot(get_structure_function_type(order, mode))
end

"""
    order(sf::AbstractStructureFunctionType)

Returns the order of the structure function.
"""
order(::ProjectedStructureFunctionType{NL, NT}) where {NL, NT} = NL + NT
order(::SecondOrderStructureFunctionType) = 2
order(::ThirdOrderStructureFunctionType) = 3
order(::FullVectorStructureFunctionType{NF}) where {NF} = NF
order(::TransverseComponentSecondOrderStructureFunctionType) = 2
order(::LongitudinalTransverseComponentThirdOrderStructureFunctionType) = 3
order(::RotationalSecondOrderStructureFunctionType) = 2
order(::DivergentSecondOrderStructureFunctionType) = 2
order(::HelmholtzDecomposition2DType) = 2
order(::ScalarStructureFunctionType{P}) where {P} = P
order(::MixedStructureFunctionType{NL, NT, P}) where {NL, NT, P} = NL + NT + P
order(::MomentTensorOperator{P}) where {P} = P
order(::ScalarDotStructureFunctionType) = 2
order(::VectorDotStructureFunctionType) = 2

"""
    scalar_order(sf) -> Int

The total power of scalar-field increments in `sf`.

Reading a pair from its other end leaves every vector quantity unchanged — `δu`, `r̂` and `n̂` all
flip together — and negates each scalar increment, so `sf` changes sign under that reading exactly
when this is odd. Such an operator is evaluated with the pair read from the lower to the upper end
along the first coordinate that separates its two points (on a sphere: from south to north, and along
one parallel from west to east); where neither end comes first, the two readings are averaged and the
moment is zero.
"""
scalar_order(::AbstractStructureFunctionType) = 0
scalar_order(::ScalarStructureFunctionType{P}) where {P} = P
scalar_order(::MixedStructureFunctionType{NL, NT, P}) where {NL, NT, P} = P
scalar_order(::ScalarDotStructureFunctionType) = 2

"""Whether `sf` changes sign when a pair is read from its other end."""
@inline is_odd_in_scalars(sf::AbstractStructureFunctionType) = isodd(scalar_order(sf))
@inline is_odd_in_scalars(::MomentTensorOperator{P}) where {P} = isodd(P)

# ---------------------------------------------------------------------------
# Polynomial contract: an operator as a contraction of the increment moment tensor
# ---------------------------------------------------------------------------

"""
    is_polynomial_operator(sf) -> Bool

Whether `sf(δu, r̂)` is a homogeneous polynomial in the packed increment `δu` with coefficients that
depend only on `r̂`, so that `Σ_pairs sf(δu, r̂)` is [`moment_contract`](@ref) of the increment moment
tensor.
"""
@inline is_polynomial_operator(::AbstractStructureFunctionType) = false
@inline is_polynomial_operator(::SecondOrderStructureFunctionType) = true
@inline is_polynomial_operator(::ProjectedStructureFunctionType{NL, NT}) where {NL, NT} = NL + NT >= 1
@inline is_polynomial_operator(::ThirdOrderStructureFunctionType) = true
@inline is_polynomial_operator(::FullVectorStructureFunctionType{NF}) where {NF} = NF >= 2 && iseven(NF)
@inline is_polynomial_operator(::TransverseComponentSecondOrderStructureFunctionType) = true
@inline is_polynomial_operator(::LongitudinalTransverseComponentThirdOrderStructureFunctionType) = true
@inline is_polynomial_operator(::ScalarStructureFunctionType{P}) where {P} = P >= 1
@inline is_polynomial_operator(::MixedStructureFunctionType{NL, NT, P}) where {NL, NT, P} =
    iseven(NT) && NL + NT + P >= 1
@inline is_polynomial_operator(::ScalarDotStructureFunctionType) = true
@inline is_polynomial_operator(::VectorDotStructureFunctionType) = true
@inline is_polynomial_operator(::MomentTensorOperator) = true

"""
    SymmetricMoments{W, P}(data::SVector)

The rank-`P` symmetric increment moment tensor over `W` packed components, stored as its
`binomial(W + P - 1, P)` independent entries `M[j₁ ≤ … ≤ j_P]` in the order of
[`symmetric_indices`](@ref).
"""
struct SymmetricMoments{W, P, N, T}
    data::SA.SVector{N, T}
    function SymmetricMoments{W, P}(data::SA.SVector{N, T}) where {W, P, N, T}
        N == binomial(W + P - 1, P) || throw(DimensionMismatch(
            "a rank-$P symmetric tensor over $W components has $(binomial(W + P - 1, P)) " *
            "independent entries; got $N",
        ))
        return new{W, P, N, T}(data)
    end
end

Base.eltype(::SymmetricMoments{W, P, N, T}) where {W, P, N, T} = T

"""
    symmetric_indices(Val(W), Val(P)) -> NTuple{N, NTuple{P, Int}}

The sorted multi-indices `j₁ ≤ … ≤ j_P` over `1:W` in storage order: entry `n` has
`n - 1 = Σ_k binomial(j_k + k - 2, k)`.
"""
@generated function symmetric_indices(::Val{W}, ::Val{P}) where {W, P}
    combos = [c for c in Iterators.product(ntuple(_ -> 1:W, P)...) if issorted(c)]
    sort!(combos; by = c -> sum(binomial(c[k] + k - 2, k) for k in 1:P))
    return Meta.quot(Tuple(combos))
end

# Storage position of a multi-index in any order: sort it, then rank the sorted tuple.
@generated function symmetric_rank(::Val{W}, ::Val{P}, idx::NTuple{P, Int}) where {W, P}
    vars = [Symbol(:j, k) for k in 1:P]
    ex = Expr[]
    for k in 1:P
        push!(ex, :($(vars[k]) = idx[$k]))
    end
    for k in 2:P, m in k:-1:2
        a, b = vars[m - 1], vars[m]
        push!(ex, :(if $a > $b; $a, $b = $b, $a; end))
    end
    terms = Expr[]
    for k in 1:P
        lut = ntuple(n -> binomial(n - 1, k), W + P)
        push!(terms, :($(lut)[$(vars[k]) + $(k - 1)]))
    end
    push!(ex, :(return 1 + $(Expr(:call, :+, terms...))))
    return Expr(:block, ex...)
end

"""
    moment_contract(sf, M::SymmetricMoments, r̂, ::Val{V}, ::Val{K})

`Σ_pairs sf(δu, r̂)` from the increment moment tensor `M[i₁, …, i_p] = Σ_pairs Π_k δu[i_k]` of the
pairs sharing the unit separation `r̂`. `δu` is packed as `V` vector fields of width `D = length(r̂)`
followed by `K` scalar fields, so `M` has `W = V * D + K` components. Operators that are not
polynomials in `δu` have no method.
"""
function moment_contract(sf::AbstractStructureFunctionType, M, r̂, ::Val, ::Val)
    throw(ArgumentError(
        "$(typeof(sf)) is not a polynomial in the increment, so its pair average is not a " *
        "contraction of an increment moment tensor. Omit the spectral backend for the lag sweep, " *
        "which evaluates any pairwise operator exactly.",
    ))
end

@inline function _check_packed_width(::SymmetricMoments{W}, ::Val{D}, ::Val{V}, ::Val{K}) where {W, D, V, K}
    W == V * D + K || throw(DimensionMismatch(
        "the moment tensor has $W components but the field packs $V vector field(s) of width $D " *
        "and $K scalar field(s), $(V * D + K) components",
    ))
    return nothing
end

@inline _unit(::Val{W}, i::Integer, ::Type{T}) where {W, T} =
    SA.SVector{W, T}(ntuple(j -> T(j == i), Val(W)))

# `r̂` placed in vector field `a` of the packed layout, zero elsewhere.
@inline function _embed(::Val{W}, r̂::SA.SVector{D, T}, a::Integer) where {W, D, T}
    off = (a - 1) * D
    return SA.SVector{W, T}(ntuple(j -> off < j <= off + D ? r̂[j - off] : zero(T), Val(W)))
end

# One literal each: a device kernel compiles a throw of a constant, never a formatted or joined string.
# The entry points name the field and the field's layout before any moment is contracted.
@inline function _require_vector_field(a::Integer, ::Val{V}) where {V}
    1 <= a <= V || throw(ArgumentError(
        "the operator reads a vector field the field does not carry; build the field with Fields(vectors = (...), ...) or use a scalar operator",
    ))
    return nothing
end

@inline function _require_scalar_field(k::Integer, ::Val{K}) where {K}
    1 <= k <= K || throw(ArgumentError(
        "the operator reads a scalar field the field does not carry; build the field with Fields(scalars = (...), ...) to carry one",
    ))
    return nothing
end

# Σ over every index tuple of M[i₁, …, i_p] · Π_k vs[k][i_k].
@generated function _contract(
    M::SymmetricMoments{W, P, N, TM}, vs::NTuple{P, SA.SVector{W}},
) where {W, P, N, TM}
    T = P == 0 ? TM : promote_type(TM, eltype(vs.parameters[1]))
    idx = [Symbol(:i, k) for k in 1:P]
    coeff = Expr(:call, :*, (:(vs[$k][$(idx[k])]) for k in 1:P)...)
    body = :(acc += M.data[symmetric_rank(Val($W), Val($P), ($(idx...),))] * $coeff)
    for k in P:-1:1
        body = :(for $(idx[k]) in 1:$W; $body; end)
    end
    return quote
        acc = zero($T)
        @inbounds $body
        return acc
    end
end

@inline function _vector_dot(M::SymmetricMoments{W, 2}, a::Integer, b::Integer, ::Val{D}) where {W, D}
    acc = zero(eltype(M))
    @inbounds for d in 1:D
        acc += M.data[symmetric_rank(Val(W), Val(2), ((a - 1) * D + d, (b - 1) * D + d))]
    end
    return acc
end

# `n` trailing slot pairs, each Σ_c e_c ⊗ e_c over the components of vector field `a`: ‖δu‖² per pair.
@inline _norm2_pairs(M, prefix::Tuple, ::Val{0}, a, ::Val{D}, ::Val{W}, ::Type{T}) where {D, W, T} =
    _contract(M, prefix)
@inline function _norm2_pairs(
    M, prefix::Tuple, ::Val{n}, a, ::Val{D}, ::Val{W}, ::Type{T},
) where {n, D, W, T}
    acc = zero(T)
    for c in 1:D
        e = _unit(Val(W), (a - 1) * D + c, T)
        acc += _norm2_pairs(M, (prefix..., e, e), Val(n - 1), a, Val(D), Val(W), T)
    end
    return acc
end

# `n` trailing slot pairs, each Σ_c e_c ⊗ e_c − rp ⊗ rp: ‖δu_T‖² per pair.
@inline _transverse_pairs(
    M, prefix::Tuple, ::Val{0}, rp, a, ::Val{D}, ::Val{W}, ::Type{T},
) where {D, W, T} = _contract(M, prefix)
@inline function _transverse_pairs(
    M, prefix::Tuple, ::Val{n}, rp, a, ::Val{D}, ::Val{W}, ::Type{T},
) where {n, D, W, T}
    acc = zero(T)
    for c in 1:D
        e = _unit(Val(W), (a - 1) * D + c, T)
        acc += _transverse_pairs(M, (prefix..., e, e), Val(n - 1), rp, a, Val(D), Val(W), T)
    end
    return acc - _transverse_pairs(M, (prefix..., rp, rp), Val(n - 1), rp, a, Val(D), Val(W), T)
end

@inline function moment_contract(
    ::SecondOrderStructureFunctionType, M::SymmetricMoments{W, 2}, r̂::SA.SVector{D}, ::Val{V}, ::Val{K},
) where {W, D, V, K}
    _check_packed_width(M, Val(D), Val(V), Val(K))
    _require_vector_field(1, Val(V))
    return _vector_dot(M, 1, 1, Val(D))
end

@generated function moment_contract(
    sf::ProjectedStructureFunctionType{NL, NT}, M::SymmetricMoments{W, P, N, TM}, r̂::SA.SVector{D, TR},
    ::Val{V}, ::Val{K},
) where {NL, NT, W, P, N, TM, D, TR, V, K}
    NL + NT >= 1 || return :(throw(ArgumentError(
        "ProjectedStructureFunctionType{0, 0} is the constant 1 and has no moment tensor",
    )))
    T = promote_type(TM, TR)
    rps = fill(:rp, NL)
    if NT == 0
        core = :(_contract(M, ($(rps...),)))
        frame = :(rp = _embed(Val($W), r̂, 1))
    elseif NT == 2
        core = :(_transverse_pairs(M, ($(rps...),), Val(1), rp, 1, Val($D), Val($W), $T))
        frame = :(rp = _embed(Val($W), r̂, 1))
    else
        core = :(_contract(M, ($(rps...), $(fill(:np, NT)...))))
        frame = quote
            rp = _embed(Val($W), r̂, 1)
            np = _embed(Val($W), SFH.transverse_basis_vector(r̂, sf.basis), 1)
        end
    end
    return quote
        _check_packed_width(M, Val($D), Val($V), Val($K))
        _require_vector_field(1, Val($V))
        $frame
        return $core
    end
end

@inline function moment_contract(
    ::ThirdOrderStructureFunctionType, M::SymmetricMoments{W, 3}, r̂::SA.SVector{D, TR}, ::Val{V}, ::Val{K},
) where {W, D, TR, V, K}
    _check_packed_width(M, Val(D), Val(V), Val(K))
    _require_vector_field(1, Val(V))
    T = promote_type(eltype(M), TR)
    rp = _embed(Val(W), r̂, 1)
    return _norm2_pairs(M, (rp,), Val(1), 1, Val(D), Val(W), T)
end

@generated function moment_contract(
    ::FullVectorStructureFunctionType{NF}, M::SymmetricMoments{W, P, N, TM}, r̂::SA.SVector{D, TR},
    ::Val{V}, ::Val{K},
) where {NF, W, P, N, TM, D, TR, V, K}
    (NF >= 2 && iseven(NF)) || return :(throw(ArgumentError(
        "FullVectorStructureFunctionType{$($NF)} is ‖δu‖^$($NF), an odd power of a norm and not a " *
        "polynomial in the increment; the lag sweep evaluates it exactly.",
    )))
    T = promote_type(TM, TR)
    return quote
        _check_packed_width(M, Val($D), Val($V), Val($K))
        _require_vector_field(1, Val($V))
        return _norm2_pairs(M, (), Val($(NF ÷ 2)), 1, Val($D), Val($W), $T)
    end
end

@inline function moment_contract(
    ::TransverseComponentSecondOrderStructureFunctionType, M::SymmetricMoments, r̂::SA.SVector{D},
    ::Val{V}, ::Val{K},
) where {D, V, K}
    D > 1 || throw(ArgumentError(
        "T2ComponentSF averages over the transverse directions, of which there are none at D = 1",
    ))
    return moment_contract(ProjectedStructureFunctionType{0, 2}(), M, r̂, Val(V), Val(K)) / (D - 1)
end

@inline function moment_contract(
    ::LongitudinalTransverseComponentThirdOrderStructureFunctionType, M::SymmetricMoments,
    r̂::SA.SVector{D}, ::Val{V}, ::Val{K},
) where {D, V, K}
    D > 1 || throw(ArgumentError(
        "L1T2ComponentSF averages over the transverse directions, of which there are none at D = 1",
    ))
    return moment_contract(ProjectedStructureFunctionType{1, 2}(), M, r̂, Val(V), Val(K)) / (D - 1)
end

@inline function moment_contract(
    sf::ScalarStructureFunctionType{P}, M::SymmetricMoments{W, P}, r̂::SA.SVector{D}, ::Val{V}, ::Val{K},
) where {P, W, D, V, K}
    _check_packed_width(M, Val(D), Val(V), Val(K))
    _require_scalar_field(sf.field, Val(K))
    s = V * D + sf.field
    return @inbounds M.data[symmetric_rank(Val(W), Val(P), ntuple(_ -> s, Val(P)))]
end

@generated function moment_contract(
    sf::MixedStructureFunctionType{NL, NT, P}, M::SymmetricMoments{W, PM, N, TM}, r̂::SA.SVector{D, TR},
    ::Val{V}, ::Val{K},
) where {NL, NT, P, W, PM, N, TM, D, TR, V, K}
    iseven(NT) || return :(throw(ArgumentError(
        "MixedStructureFunctionType{$($NL), $($NT), $($P)} raises ‖δu_T‖ to an odd power, which is " *
        "not a polynomial in the increment; the lag sweep evaluates it exactly.",
    )))
    NL + NT + P >= 1 || return :(throw(ArgumentError(
        "MixedStructureFunctionType{0, 0, 0} is the constant 1 and has no moment tensor",
    )))
    T = promote_type(TM, TR)
    slots = vcat(fill(:rp, NL), fill(:es, P))
    return quote
        _check_packed_width(M, Val($D), Val($V), Val($K))
        _require_vector_field(sf.vector_field, Val($V))
        _require_scalar_field(sf.scalar_field, Val($K))
        rp = _embed(Val($W), r̂, sf.vector_field)
        es = _unit(Val($W), $V * $D + sf.scalar_field, $T)
        return _transverse_pairs(
            M, ($(slots...),), Val($(NT ÷ 2)), rp, sf.vector_field, Val($D), Val($W), $T,
        )
    end
end

@inline function moment_contract(
    sf::ScalarDotStructureFunctionType, M::SymmetricMoments{W, 2}, r̂::SA.SVector{D}, ::Val{V}, ::Val{K},
) where {W, D, V, K}
    _check_packed_width(M, Val(D), Val(V), Val(K))
    _require_scalar_field(sf.a, Val(K))
    _require_scalar_field(sf.b, Val(K))
    return @inbounds M.data[symmetric_rank(Val(W), Val(2), (V * D + sf.a, V * D + sf.b))]
end

@inline function moment_contract(
    sf::VectorDotStructureFunctionType, M::SymmetricMoments{W, 2}, r̂::SA.SVector{D}, ::Val{V}, ::Val{K},
) where {W, D, V, K}
    _check_packed_width(M, Val(D), Val(V), Val(K))
    _require_vector_field(sf.a, Val(V))
    _require_vector_field(sf.b, Val(V))
    return _vector_dot(M, sf.a, sf.b, Val(D))
end

end # module
