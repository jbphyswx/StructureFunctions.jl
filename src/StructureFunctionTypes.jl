module StructureFunctionTypes

using LinearAlgebra: norm
using ..HelperFunctions

abstract type AbstractStructureFunctionType end

using LoopVectorization

# norm2_reg(x) = norm(x)^2 # faster? NO
# norm_alt(x) = sqrt(sum(val->val^2,x)) # faster? NO

# # norm2_alt(x) = sum(val->val^2,x) # faster? NO
# squared = val->val^2
# norm2_alt(x) = @tturbo sum(squared,x) # faster? NO


@inline @fastmath @inbounds norm2(x) = begin # no allocations, is a lil faster (for only 2 dimensions not much faster but)
    out::eltype(x) = zero(eltype(x))
    @tturbo warn_check_args=false for i in eachindex(x)
        @inbounds @fastmath out +=  x[i]^2
    end
    return out
end

# # Something like this could reduce allocations further but pos[1] doesn't work w/ llvm and idk how else to do it, and it doesn't save thaaat much time I suppose
# @inline @fastmath @inbounds norm2!(x, pos) = begin # no allocations, is a lil faster (for only 2 dimensions not much faster but)
#     for i in eachindex(x)
#         @inbounds @fastmath pos[] +=  x[i]^2
#     end
# end


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

# The anonymous fcns might be slow and recompile every time, put them outside the struct for precompilation/consstruction only once once

# 2nd Order
# @inbounds SecondOrderStructureFunction_method(δu, r̂, pos) = norm2!(δu, pos) # if we could figure out an inplace solution we could reduce allocations a lot, but harder for other types of SFs...
@inbounds SecondOrderStructureFunction_method(δu, r̂) = norm2(δu) 

struct SecondOrderStructureFunction <: AbstractStructureFunctionType # D2 |δu|² = δu·δu = = δu_l·δu_l + δu_t·δu_t = δu_l² + δu_t² = D2l + D2t
    method::Function
    SecondOrderStructureFunction() = new(SecondOrderStructureFunction_method) # norm2 is slightly faster but isn't coming through in full runs for some reason...
    # SecondOrderStructureFunction() = new((δu, r̂) -> norm(δu)^2)
end

@inbounds LongitudinalSecondOrderStructureFunction_method(δu, r̂) = mδu_l(δu, r̂)^2
struct LongitudinalSecondOrderStructureFunction <: AbstractStructureFunctionType # D2L |δu_l|² = u_l·δu_l
    method::Function
    LongitudinalSecondOrderStructureFunction() = new(LongitudinalSecondOrderStructureFunction_method)
end

@inbounds TransverseSecondOrderStructureFunction_method(δu, r̂) = norm2(δu_t(δu, r̂))  # maybe faster than mδu_l(δu, r̂)^2 since we never calculate n̂
struct TransverseSecondOrderStructureFunction <: AbstractStructureFunctionType # D2T |δu_t|² = u_t·δu_t
    method::Function
    TransverseSecondOrderStructureFunction() = new(TransverseSecondOrderStructureFunction_method)
end

struct RotationalSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented
struct DivergentSecondOrderStructureFunction <: AbstractStructureFunctionType end # NotImplemented


# 3rd Order
# Note, we use mδu_{} instead of |δu_{}| because the magnitudes have a sign (relative to normal vector)! (I think this is the right way to do it, see https://doi.org/10.1002/2016GL069405 and old Lindberg and Cho papers)

ThirdOrderStructureFunction_method(δu, r̂) = norm(δu)^3  # i think sqrt and then ³ is fastest since we have to calculate sqrt qnyway
struct ThirdOrderStructureFunction <: AbstractStructureFunctionType # |δu|³  = |δu| δu·δu
    method::Function
    ThirdOrderStructureFunction() = new(ThirdOrderStructureFunction_method)
end

DiagonalConsistentThirdOrderStructureFunction_method(δu, r̂) = mδu_l(δu, r̂)^3
struct DiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_l³ 
    method::Function
    DiagonalConsistentThirdOrderStructureFunction() = new(DiagonalConsistentThirdOrderStructureFunction_method)
end

DiagonalInconsistentThirdOrderStructureFunction_method(δu, r̂) = mδu_l(δu, r̂)^2 * mδu_t(δu, r̂)  # you could probably cache some intermediate results here
struct DiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # |δu_l|² * mδu_t
    method::Function
    DiagonalInconsistentThirdOrderStructureFunction() = new(DiagonalInconsistentThirdOrderStructureFunction_method )
end

OffDiagonalInconsistentThirdOrderStructureFunction_method(δu, r̂) = mδu_l(δu, r̂) * norm2(δu_t(δu, r̂))  # norm(δu_t) might be faster than mδu_t(δu, r̂)^2 since we never calculate n̂ and we're squaring anyway so sign is irrelevant
struct OffDiagonalInconsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_l * |δu_t|²
    method::Function
    OffDiagonalInconsistentThirdOrderStructureFunction() = new(OffDiagonalInconsistentThirdOrderStructureFunction_method) 
end

OffDiagonalConsistentThirdOrderStructureFunction_method(δu, r̂) = mδu_t(δu, r̂)^3  # you could probably cache some intermediate results here
struct OffDiagonalConsistentThirdOrderStructureFunction <: AbstractStructureFunctionType # mδu_t³
    method::Function
    OffDiagonalConsistentThirdOrderStructureFunction() = new(OffDiagonalConsistentThirdOrderStructureFunction_method)
end



end
