module StructureFunctionTypes

using LinearAlgebra: LinearAlgebra as LA
using ..HelperFunctions: HelperFunctions as SFH

abstract type AbstractStructureFunctionType end

# Identity call for backward compatibility: Allows SFType() where SFType is now a constant instance.
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
struct ProjectedStructureFunctionType{NL,NT} <: AbstractStructureFunctionType end

const ProjectedStructureFunction = ProjectedStructureFunctionType # Longhand alias for the type

ProjectedStructureFunctionType(NL::Integer, NT::Integer) = ProjectedStructureFunctionType{NL,NT}()

"""
    (sf::ProjectedStructureFunctionType{NL,NT})(δu, r̂)

Compute the structure function kernel for longitudinal/transverse components.

- `NL` : power of longitudinal component δu_l
- `NT` : power of transverse component ||δu_t||
"""
@generated function (sf::ProjectedStructureFunctionType{NL,NT})(δu, r̂) where {NL,NT}
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
            # fast path: sum-of-squares, no sqrt
            ex = :($ex * norm2(SFH.δu_t(δu, r̂)))
        elseif NT == 3
            # standard magnitude cubed (precomputed scalar), avoids fractional exponent
            ex = :($ex * (SFH.mδu_t(δu, r̂)^3))
        elseif NT == 1
            ex = :($ex * SFH.mδu_t(δu, r̂))
        else
            # fallback: use scalar magnitude raised to NT (for uncommon powers)
            ex = :($ex * (SFH.mδu_t(δu, r̂)^$NT))
        end
    end

    return ex
end

# ---------------------------------------------------------------------------
"""
    FullVectorStructureFunctionType{NF}

Parametric type representing structure function operators
with only the **full vector magnitude** ||δu||.
"""
struct FullVectorStructureFunctionType{NF} <: AbstractStructureFunctionType end

const FullVectorStructureFunction = FullVectorStructureFunctionType # Longhand alias for the type

FullVectorStructureFunctionType(NF::Integer) = FullVectorStructureFunctionType{NF}()

"""
    (sf::FullVectorStructureFunctionType{NF})(δu, r̂)

Compute the structure function kernel for the full vector magnitude:

- `NF` : power of ||δu||
"""
@generated function (sf::FullVectorStructureFunctionType{NF})(δu, r̂) where {NF}
    if NF == 2
        return :(norm2(δu))
    else
        return :(LA.norm(δu)^$NF)
    end
end

# ---------------------------------------------------------------------------
# Named Constants: Type Aliases (longhand and shorthands)

# 2nd order Type Aliases
const SecondOrderStructureFunctionType             = FullVectorStructureFunctionType{2}
const LongitudinalSecondOrderStructureFunctionType = ProjectedStructureFunctionType{2,0}
const TransverseSecondOrderStructureFunctionType   = ProjectedStructureFunctionType{0,2}

# 3rd order Type Aliases
const ThirdOrderStructureFunctionType                 = FullVectorStructureFunctionType{3}
const DiagonalConsistentThirdOrderStructureFunctionType  = ProjectedStructureFunctionType{3,0}
const DiagonalInconsistentThirdOrderStructureFunctionType = ProjectedStructureFunctionType{2,1}
const OffDiagonalInconsistentThirdOrderStructureFunctionType = ProjectedStructureFunctionType{1,2}
const OffDiagonalConsistentThirdOrderStructureFunctionType = ProjectedStructureFunctionType{0,3}

# Shorthand Type Aliases
const L2SFType = LongitudinalSecondOrderStructureFunctionType
const T2SFType = TransverseSecondOrderStructureFunctionType
const L3SFType = DiagonalConsistentThirdOrderStructureFunctionType

# ---------------------------------------------------------------------------
# Named Constants: Singleton Functors (The "Longhand" names now point to instances)

# 2nd order Singleton Functors
const SecondOrderStructureFunction             = SecondOrderStructureFunctionType()
const LongitudinalSecondOrderStructureFunction   = LongitudinalSecondOrderStructureFunctionType()
const TransverseSecondOrderStructureFunction     = TransverseSecondOrderStructureFunctionType()
const TransverseStructureFunction                = TransverseSecondOrderStructureFunction # Legacy alias

# 3rd order Singleton Functors
const ThirdOrderStructureFunction                 = ThirdOrderStructureFunctionType()
const DiagonalConsistentThirdOrderStructureFunction  = DiagonalConsistentThirdOrderStructureFunctionType()
const DiagonalInconsistentThirdOrderStructureFunction = DiagonalInconsistentThirdOrderStructureFunctionType()
const OffDiagonalInconsistentThirdOrderStructureFunction = OffDiagonalInconsistentThirdOrderStructureFunctionType()
const OffDiagonalConsistentThirdOrderStructureFunction = OffDiagonalConsistentThirdOrderStructureFunctionType()

# Shorthand Singleton Functors
const L2SF = LongitudinalSecondOrderStructureFunction
const T2SF = TransverseSecondOrderStructureFunction
const L3SF = DiagonalConsistentThirdOrderStructureFunction

# Compatibility layer for Rotational/Divergent (placeholders)
struct RotationalSecondOrderStructureFunctionType <: AbstractStructureFunctionType end
struct DivergentSecondOrderStructureFunctionType <: AbstractStructureFunctionType end

const RotationalSecondOrderStructureFunction = RotationalSecondOrderStructureFunctionType()
const DivergentSecondOrderStructureFunction = DivergentSecondOrderStructureFunctionType()

# ---------------------------------------------------------------------------
# Convenience Mappings

const SF_TYPE_MAP = Dict{Symbol, AbstractStructureFunctionType}(
    :SecondOrderStructureFunction => SecondOrderStructureFunction,
    :LongitudinalSecondOrderStructureFunction => LongitudinalSecondOrderStructureFunction,
    :TransverseSecondOrderStructureFunction => TransverseSecondOrderStructureFunction,
    :TransverseStructureFunction => TransverseStructureFunction,
    :RotationalSecondOrderStructureFunction => RotationalSecondOrderStructureFunction,
    :DivergentSecondOrderStructureFunction => DivergentSecondOrderStructureFunction,
    :ThirdOrderStructureFunction => ThirdOrderStructureFunction,
    :DiagonalConsistentThirdOrderStructureFunction => DiagonalConsistentThirdOrderStructureFunction,
    :DiagonalInconsistentThirdOrderStructureFunction => DiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalInconsistentThirdOrderStructureFunction => OffDiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalConsistentThirdOrderStructureFunction => OffDiagonalConsistentThirdOrderStructureFunction,
    :L2SF => L2SF,
    :T2SF => T2SF,
    :L3SF => L3SF,
)

export AbstractStructureFunctionType,
    SecondOrderStructureFunctionType,
    LongitudinalSecondOrderStructureFunctionType,
    TransverseSecondOrderStructureFunctionType,
    ThirdOrderStructureFunctionType,
    DiagonalConsistentThirdOrderStructureFunctionType,
    DiagonalInconsistentThirdOrderStructureFunctionType,
    OffDiagonalInconsistentThirdOrderStructureFunctionType,
    OffDiagonalConsistentThirdOrderStructureFunctionType,
    RotationalSecondOrderStructureFunctionType,
    DivergentSecondOrderStructureFunctionType,
    L2SFType, T2SFType, L3SFType,
    
    SecondOrderStructureFunction,
    LongitudinalSecondOrderStructureFunction,
    TransverseSecondOrderStructureFunction,
    TransverseStructureFunction,
    RotationalSecondOrderStructureFunction,
    DivergentSecondOrderStructureFunction,
    ThirdOrderStructureFunction,
    DiagonalConsistentThirdOrderStructureFunction,
    DiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalConsistentThirdOrderStructureFunction,
    L2SF, T2SF, L3SF,
    
    ProjectedStructureFunctionType,
    FullVectorStructureFunctionType,
    ProjectedStructureFunction,
    FullVectorStructureFunction,
    get_structure_function_type

get_structure_function_type(x::String) = get_structure_function_type(Symbol(x))
function get_structure_function_type(x::Symbol)
    if haskey(SF_TYPE_MAP, x)
        return SF_TYPE_MAP[x]
    else
        error("Unknown structure function type: $x")
    end
end

"""
    order(sf::AbstractStructureFunctionType)

Returns the order of the structure function.
"""
order(::ProjectedStructureFunctionType{NL,NT}) where {NL,NT} = NL + NT
order(::FullVectorStructureFunctionType{NF}) where {NF} = NF
order(::RotationalSecondOrderStructureFunctionType) = 2
order(::DivergentSecondOrderStructureFunctionType) = 2

end # module
