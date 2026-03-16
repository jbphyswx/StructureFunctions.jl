module StructureFunctionTypes

using LinearAlgebra: norm
using ..HelperFunctions

abstract type AbstractStructureFunctionType end

using LoopVectorization

norm2_reg(x) = norm(x)^2 # faster? NO
norm_alt(x) = sqrt(sum(val->val^2,x)) # faster? NO

# norm2_alt(x) = sum(val->val^2,x) # faster? NO
squared = val->val^2
norm2_alt(x) = @tturbo sum(squared,x) # faster? NO


@inline @fastmath @inbounds norm2(x) = begin # no allocations, is a lil faster (for only 2 dimensions not much faster but)
    out::eltype(x) = zero(eltype(x))
    @tturbo warn_check_args=false for i in eachindex(x)
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

get_structure_function_type(x::String) = get_structure_function_type(Symbol(x)) # methods to get structure function instance from string or symbol
get_structure_function_type(x::Symbol) = eval(x)()

# see https://doi.org/10.1002/2016GL069405 for definitions

# The anonymous fcns might be slow...

# 2nd Order
struct SecondOrderStructureFunction <: AbstractStructureFunctionType # D2 |δu|² = δu·δu = = δu_l·δu_l + δu_t·δu_t = δu_l² + δu_t² = D2l + D2t
    method::Function
    SecondOrderStructureFunction() = new((δu, r̂) -> norm2(δu)) # norm2 is slightly faster but isn't coming through in full runs for some reason...
    # SecondOrderStructureFunction() = new((δu, r̂) -> norm(δu)^2)
end

struct LongitudinalSecondOrderStructureFunction <: AbstractStructureFunctionType # D2L |δu_l|² = u_l·δu_l
    method::Function
    @inbounds LongitudinalSecondOrderStructureFunction() = new((δu, r̂) -> mδu_l(δu, r̂)^2 )
end

struct TransverseSecondOrderStructureFunction <: AbstractStructureFunctionType # D2T |δu_t|² = u_t·δu_t
    method::Function
    TransverseSecondOrderStructureFunction() = new((δu, r̂) -> norm2(δu_t(δu, r̂))) # maybe faster than mδu_l(δu, r̂)^2 since we never calculate n̂
end

struct RotationalSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented
struct DivergentSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented


# 3rd Order
# Note, we use mδu_{} instead of |δu_{}| because the magnitudes have a sign (relative to normal vector)! (I think this is the right way to do it, see https://doi.org/10.1002/2016GL069405 and old Lindberg and Cho papers)

struct ThirdOrderStructureFunction <: AbstractStructureFunctionType # |δu|³  = |δu| δu·δu
    method::Function
    ThirdOrderStructureFunction() = new((δu, r̂) -> norm(δu)^3) # i think sqrt and then ³ is fastest since we have to calculate sqrt qnyway
end

struct DiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_l³ 
    method::Function
    DiagonalConsistentThirdOrderStructureFunction() = new((δu, r̂) -> mδu_l(δu, r̂)^3 )
end

struct DiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # |δu_l|² * mδu_t
    method::Function
    DiagonalInconsistentThirdOrderStructureFunction() = new((δu, r̂) -> mδu_l(δu, r̂)^2 * mδu_t(δu, r̂) ) # you could probably cache some intermediate results here
end

struct OffDiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_l * |δu_t|²
    method::Function
    OffDiagonalInconsistentThirdOrderStructureFunction() = new((δu, r̂) -> mδu_l(δu, r̂) * norm2(δu_t(δu, r̂))) # norm(δu_t) might be faster than mδu_t(δu, r̂)^2 since we never calculate n̂ and we're squaring anyway so sign is irrelevant
end

struct OffDiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_t³
    method::Function
    OffDiagonalConsistentThirdOrderStructureFunction() = new((δu, r̂) -> mδu_t(δu, r̂)^3 ) # you could probably cache some intermediate results here
end



end
