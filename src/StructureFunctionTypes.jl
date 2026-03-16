module StructureFunctionTypes

using LinearAlgebra: LinearAlgebra as LA
import ..HelperFunctions: HelperFunctions as HF

abstract type AbstractStructureFunctionType end

using LoopVectorization: LoopVectorization as LV # TODO: Move to extension or replace with Polyester/SIMD as part of modernization (Phase 3/4)

@inline @fastmath @inbounds norm2(x) = begin
    out = zero(eltype(x))
    for i in eachindex(x)
        @inbounds @fastmath out +=  x[i]^2
    end
    return out
end

export SecondOrderStructureFunction,
    LongitudinalSecondOrderStructureFunction,
    TransverseSecondOrderStructureFunction,
    RotationalSecondOrderStructureFunction,
    DivergentSecondOrderStructureFunction,
    ThirdOrderStructureFunction,
    DiagonalConsistentThirdOrderStructureFunction,
    DiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction,
    OffDiagonalConsistentThirdOrderStructureFunction,
    get_structure_function_type

# see https://doi.org/10.1002/2016GL069405 for definitions

# 2nd Order
struct SecondOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::SecondOrderStructureFunction)(δu, r̂) = norm2(δu)

struct LongitudinalSecondOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::LongitudinalSecondOrderStructureFunction)(δu, r̂) = HF.mδu_l(δu, r̂)^2

struct TransverseSecondOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::TransverseSecondOrderStructureFunction)(δu, r̂) = norm2(HF.δu_t(δu, r̂))

struct RotationalSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented
struct DivergentSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented


# 3rd Order
struct ThirdOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::ThirdOrderStructureFunction)(δu, r̂) = LA.norm(δu)^3

struct DiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::DiagonalConsistentThirdOrderStructureFunction)(δu, r̂) = HF.mδu_l(δu, r̂)^3

struct DiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::DiagonalInconsistentThirdOrderStructureFunction)(δu, r̂) = HF.mδu_l(δu, r̂)^2 * HF.mδu_t(δu, r̂)

struct OffDiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::OffDiagonalInconsistentThirdOrderStructureFunction)(δu, r̂) = HF.mδu_l(δu, r̂) * norm2(HF.δu_t(δu, r̂))

struct OffDiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType end
@inline (::OffDiagonalConsistentThirdOrderStructureFunction)(δu, r̂) = HF.mδu_t(δu, r̂)^3


# Mapping for get_structure_function_type to avoid eval. Convenience function for end users.
const SF_TYPE_MAP = Dict{Symbol, Type{<:AbstractStructureFunctionType}}(
    :SecondOrderStructureFunction => SecondOrderStructureFunction,
    :LongitudinalSecondOrderStructureFunction => LongitudinalSecondOrderStructureFunction,
    :TransverseSecondOrderStructureFunction => TransverseSecondOrderStructureFunction,
    :RotationalSecondOrderStructureFunction => RotationalSecondOrderStructureFunction,
    :DivergentSecondOrderStructureFunction => DivergentSecondOrderStructureFunction,
    :ThirdOrderStructureFunction => ThirdOrderStructureFunction,
    :DiagonalConsistentThirdOrderStructureFunction => DiagonalConsistentThirdOrderStructureFunction,
    :DiagonalInconsistentThirdOrderStructureFunction => DiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalInconsistentThirdOrderStructureFunction => OffDiagonalInconsistentThirdOrderStructureFunction,
    :OffDiagonalConsistentThirdOrderStructureFunction => OffDiagonalConsistentThirdOrderStructureFunction,
)

get_structure_function_type(x::String) = get_structure_function_type(Symbol(x))
function get_structure_function_type(x::Symbol)
    if haskey(SF_TYPE_MAP, x)
        return SF_TYPE_MAP[x]()
    else
        error("Unknown structure function type: $x")
    end
end

end
