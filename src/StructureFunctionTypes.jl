module StructureFunctionTypes

using LinearAlgebra: norm
using ..HelperFunctions

abstract type AbstractStructureFunctionType end


export SecondOrderStructureFunction,
    LongitudinalSecondOrderStructureFunction,
    TransverseSecondOrderStructureFunction,
    RotationalSecondOrderStructureFunction,
    DivergenceSecondOrderStructureFunction,
    ThirdOrderStructureFunction,
    DiagonalConsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction,
    get_structure_function_type

get_structure_function_type(x::String) = get_structure_function_type(Symbol(x)) # methods to get structure function instance from string or symbol
get_structure_function_type(x::Symbol) = eval(x)()

# see https://doi.org/10.1002/2016GL069405 for definitions


# 2nd Order
struct SecondOrderStructureFunction <: AbstractStructureFunctionType # D2 |δu|² = δu·δu = = δu_l·δu_l + δu_t·δu_t = δu_l² + δu_t² = D2l + D2t
    method::Function
    SecondOrderStructureFunction() = new((δu, r̂) -> norm(δu)^2)
end

struct LongitudinalSecondOrderStructureFunction <: AbstractStructureFunctionType # D2L |δu_l|² = u_l·δu_l
    method::Function
    LongitudinalSecondOrderStructureFunction() = new((δu, r̂) -> norm(δu_l(δu, r̂))^2)
end

struct TransverseSecondOrderStructureFunction <: AbstractStructureFunctionType # D2T |δu_t|² = u_t·δu_t
    method::Function
    TransverseSecondOrderStructureFunction() = new((δu, r̂) -> norm(δu_t(δu, r̂))^2)
end

struct RotationalSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented
struct DivergenceSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented


# 3rd Order
struct ThirdOrderStructureFunction <: AbstractStructureFunctionType # |δu|³  = |δu| δu·δu
    method::Function
    ThirdOrderStructureFunction() = new((δu, r̂) -> norm(δu)^3)
end

struct DiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # δu_l³
    method::Function
    DiagonalConsistentThirdOrderStructureFunction() = new((δu, r̂) -> norm(δu_l(δu, r̂))^3)
end

struct OffDiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # δu_l**2 * δu_t
    method::Function
    OffDiagonalInconsistentThirdOrderStructureFunction() = new((δu, r̂) -> norm(δu_l(δu, r̂))^2 * norm(δu_t(δu, r̂)))
end



end
