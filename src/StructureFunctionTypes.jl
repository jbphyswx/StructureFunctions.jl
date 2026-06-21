module StructureFunctionTypes

using LinearAlgebra: LinearAlgebra as LA
using ..HelperFunctions: HelperFunctions as SFH

abstract type AbstractStructureFunctionType end
abstract type AbstractPairwiseStructureFunctionType <: AbstractStructureFunctionType end
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
    ProjectedStructureFunctionType{NL,NT}

Parametric type representing structure function operators
with **longitudinal (`NL`)** and **transverse (`NT`)** contributions.
"""
struct ProjectedStructureFunctionType{NL, NT} <: AbstractPairwiseStructureFunctionType end

const ProjectedStructureFunction = ProjectedStructureFunctionType # Longhand alias for the type

ProjectedStructureFunctionType(NL::Integer, NT::Integer) =
    ProjectedStructureFunctionType{NL, NT}()

"""
    (sf::ProjectedStructureFunctionType{NL,NT})(δu, r̂)

Compute the structure function kernel for longitudinal/transverse components.

- `NL` : power of longitudinal component δu_l
- `NT` : power of transverse component ||δu_t||
"""
@generated function (sf::ProjectedStructureFunctionType{NL, NT})(δu, r̂) where {NL, NT}
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

    # Transverse contribution
    if !iszero(NT)
        if NT == 2
            # fast path: invariant transverse energy, no basis and no sqrt
            ex = :($ex * SFH.transverse_norm2(δu, r̂))
        elseif NT == 3
            # standard magnitude cubed (precomputed scalar), avoids fractional exponent
            ex = :($ex * (SFH.mδu_t(δu, r̂)^3))
        elseif NT == 1
            ex = :($ex * SFH.mδu_t(δu, r̂))
        else
            # generic path: use scalar magnitude raised to NT (for uncommon powers)
            ex = :($ex * (SFH.mδu_t(δu, r̂)^$NT))
        end
    end

    return ex
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

Generic full-vector norm-power operator, ``||δu||^NF``. This is not used for
`S3SF`, whose conventional definition is [`ThirdOrderStructureFunctionType`](@ref).
"""
struct FullVectorStructureFunctionType{NF} <: AbstractPairwiseStructureFunctionType end

const FullVectorStructureFunction = FullVectorStructureFunctionType

FullVectorStructureFunctionType(NF::Integer) = FullVectorStructureFunctionType{NF}()

@inline (::SecondOrderStructureFunctionType)(δu, r̂) = norm2(δu)

@inline (::ThirdOrderStructureFunctionType)(δu, r̂) =
    SFH.mδu_l(δu, r̂) * norm2(δu)

@generated function (::FullVectorStructureFunctionType{NF})(δu, r̂) where {NF}
    NF == 2 && return :(norm2(δu))
    return :(LA.norm(δu)^$NF)
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
    SFH.transverse_component_norm2(δu, r̂)

@inline (::LongitudinalTransverseComponentThirdOrderStructureFunctionType)(δu, r̂) =
    SFH.mδu_l(δu, r̂) * SFH.transverse_component_norm2(δu, r̂)

# ---------------------------------------------------------------------------
# Named Constants: Type Aliases (longhand and shorthands)

# 2nd order Type Aliases
const LongitudinalSecondOrderStructureFunctionType = ProjectedStructureFunctionType{2, 0}
const TransverseSecondOrderStructureFunctionType = ProjectedStructureFunctionType{0, 2}
const T2ComponentSFType = TransverseComponentSecondOrderStructureFunctionType

# 3rd order Type Aliases
const DiagonalConsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{3, 0}
const DiagonalInconsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{2, 1}
const OffDiagonalInconsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{1, 2}
const OffDiagonalConsistentThirdOrderStructureFunctionType =
    ProjectedStructureFunctionType{0, 3}

# Shorthand Type Aliases
const S2SFType = SecondOrderStructureFunctionType
const L2SFType = LongitudinalSecondOrderStructureFunctionType
const T2SFType = TransverseSecondOrderStructureFunctionType
const S3SFType = ThirdOrderStructureFunctionType
const L3SFType = DiagonalConsistentThirdOrderStructureFunctionType
const T3SFType = OffDiagonalConsistentThirdOrderStructureFunctionType
const L2T1SFType = DiagonalInconsistentThirdOrderStructureFunctionType
const L1T2SFType = OffDiagonalInconsistentThirdOrderStructureFunctionType
const L1T2ComponentSFType = LongitudinalTransverseComponentThirdOrderStructureFunctionType

# ---------------------------------------------------------------------------
# Named Constants: Singleton Functors (The "Longhand" names now point to instances)

# 2nd order Singleton Functors
const SecondOrderStructureFunction = SecondOrderStructureFunctionType()
const LongitudinalSecondOrderStructureFunction =
    LongitudinalSecondOrderStructureFunctionType()
const TransverseSecondOrderStructureFunction = TransverseSecondOrderStructureFunctionType()
const T2ComponentSF = TransverseComponentSecondOrderStructureFunctionType()

# 3rd order Singleton Functors
const ThirdOrderStructureFunction = ThirdOrderStructureFunctionType()
const DiagonalConsistentThirdOrderStructureFunction =
    DiagonalConsistentThirdOrderStructureFunctionType()
const DiagonalInconsistentThirdOrderStructureFunction =
    DiagonalInconsistentThirdOrderStructureFunctionType()
const OffDiagonalInconsistentThirdOrderStructureFunction =
    OffDiagonalInconsistentThirdOrderStructureFunctionType()
const OffDiagonalConsistentThirdOrderStructureFunction =
    OffDiagonalConsistentThirdOrderStructureFunctionType()
const L1T2ComponentSF = LongitudinalTransverseComponentThirdOrderStructureFunctionType()

# Shorthand Singleton Functors
const S2SF = SecondOrderStructureFunction
const L2SF = LongitudinalSecondOrderStructureFunction
const T2SF = TransverseSecondOrderStructureFunction
const S3SF = ThirdOrderStructureFunction
const L3SF = DiagonalConsistentThirdOrderStructureFunction
const T3SF = OffDiagonalConsistentThirdOrderStructureFunction
const L2T1SF = DiagonalInconsistentThirdOrderStructureFunction
const L1T2SF = OffDiagonalInconsistentThirdOrderStructureFunction

"""
    RotationalSecondOrderStructureFunctionType()

Helmholtz-derived 2D rotational second-order component. This is not a
pairwise operator; compute it from binned `L2SF`/`T2SF` with
`helmholtz_decompose_2d`.
"""
struct RotationalSecondOrderStructureFunctionType <: AbstractDerivedStructureFunctionType end

"""
    DivergentSecondOrderStructureFunctionType()

Helmholtz-derived 2D divergent second-order component. This is not a
pairwise operator; compute it from binned `L2SF`/`T2SF` with
`helmholtz_decompose_2d`.
"""
struct DivergentSecondOrderStructureFunctionType <: AbstractDerivedStructureFunctionType end

"""
    HelmholtzDecomposition2DType()

Derived quantity describing the 2D isotropic Helmholtz decomposition into
rotational and divergent second-order components.
"""
struct HelmholtzDecomposition2DType <: AbstractDerivedStructureFunctionType end

const RotationalSecondOrderStructureFunction = RotationalSecondOrderStructureFunctionType()
const DivergentSecondOrderStructureFunction = DivergentSecondOrderStructureFunctionType()
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
    T2ComponentSF,
    RotationalSecondOrderStructureFunction,
    DivergentSecondOrderStructureFunction,
    HelmholtzDecomposition2DOperator,
    ThirdOrderStructureFunction,
    DiagonalConsistentThirdOrderStructureFunction,
    DiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalConsistentThirdOrderStructureFunction,
    L1T2ComponentSF,
    S2SF, L2SF, T2SF, S3SF, L3SF, T3SF, L2T1SF, L1T2SF,
    T2ComponentSF, L1T2ComponentSF, ProjectedStructureFunctionType,
    FullVectorStructureFunctionType,
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

end # module
