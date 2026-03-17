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
    ProjectedStructureFunction{NL,NT}

Parametric type representing structure functions
with **longitudinal (`NL`)** and **transverse (`NT`)** contributions.
"""
struct ProjectedStructureFunction{NL,NT} <: AbstractStructureFunctionType end

"""
    (sf::ProjectedStructureFunction{NL,NT})(δu, r̂)

Compute the structure function for longitudinal/transverse components.

- `NL` : power of longitudinal component δu_l
- `NT` : power of transverse component ||δu_t||
"""
@generated function (sf::ProjectedStructureFunction{NL,NT})(δu, r̂) where {NL,NT}
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
    FullVectorStructureFunction{NF}

Parametric type representing structure functions
with only the **full vector magnitude** ||δu||.
"""
struct FullVectorStructureFunction{NF} <: AbstractStructureFunctionType end

"""
    (sf::FullVectorStructureFunction{NF})(δu, r̂)

Compute the structure function for the full vector magnitude:

- `NF` : power of ||δu||
"""
@generated function (sf::FullVectorStructureFunction{NF})(δu, r̂) where {NF}
    if NF == 2
        return :(norm2(δu))
    else
        return :(LA.norm(δu)^$NF)
    end
end

# ---------------------------------------------------------------------------
# Backward Compatibility & Named Constants

# 2nd order
const SecondOrderStructureFunction           = FullVectorStructureFunction{2}()
const LongitudinalSecondOrderStructureFunction = ProjectedStructureFunction{2,0}()
const TransverseStructureFunction              = ProjectedStructureFunction{0,2}() # Legacy alias
const TransverseSecondOrderStructureFunction   = ProjectedStructureFunction{0,2}()

# 3rd order
const ThirdOrderStructureFunction                 = FullVectorStructureFunction{3}()
const DiagonalConsistentThirdOrderStructureFunction  = ProjectedStructureFunction{3,0}()
const DiagonalInconsistentThirdOrderStructureFunction = ProjectedStructureFunction{2,1}()
const OffDiagonalInconsistentThirdOrderStructureFunction = ProjectedStructureFunction{1,2}()
const OffDiagonalConsistentThirdOrderStructureFunction = ProjectedStructureFunction{0,3}()

# Compatibility layer for Rotational/Divergent (placeholders)
struct RotationalSecondOrderStructureFunction <: AbstractStructureFunctionType end
struct DivergentSecondOrderStructureFunction <: AbstractStructureFunctionType end

# ---------------------------------------------------------------------------
# Convenience Mappings

const SF_TYPE_MAP = Dict{Symbol, AbstractStructureFunctionType}(
    :SecondOrderStructureFunction => SecondOrderStructureFunction,
    :LongitudinalSecondOrderStructureFunction => LongitudinalSecondOrderStructureFunction,
    :TransverseSecondOrderStructureFunction => TransverseSecondOrderStructureFunction,
    :TransverseStructureFunction => TransverseStructureFunction,
    :RotationalSecondOrderStructureFunction => RotationalSecondOrderStructureFunction(),
    :DivergentSecondOrderStructureFunction => DivergentSecondOrderStructureFunction(),
    :ThirdOrderStructureFunction => ThirdOrderStructureFunction,
    :DiagonalConsistentThirdOrderStructureFunction => DiagonalConsistentThirdOrderStructureFunction,
    :DiagonalInconsistentThirdOrderStructureFunction => DiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalInconsistentThirdOrderStructureFunction => OffDiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalConsistentThirdOrderStructureFunction => OffDiagonalConsistentThirdOrderStructureFunction,
)

export SecondOrderStructureFunction,
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
order(::ProjectedStructureFunction{NL,NT}) where {NL,NT} = NL + NT
order(::FullVectorStructureFunction{NF}) where {NF} = NF

end # module
