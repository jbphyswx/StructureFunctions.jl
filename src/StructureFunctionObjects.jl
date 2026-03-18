module StructureFunctionObjects

using ..StructureFunctionTypes: StructureFunctionTypes as SFT

export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts

"""
    AbstractStructureFunction

Abstract base type for all structure function result containers.
"""
abstract type AbstractStructureFunction end

"""
    StructureFunction{FT, OT, BT, VT}

Metadata-rich result object for structure function calculations.

- `operator`: The specific operator used (e.g., `L2SF`).
- `distance`: The coordinate container (can be bin midpoints, edges, or point distances).
- `values`: The computed structure function values.
"""
struct StructureFunction{FT, OT <: SFT.AbstractStructureFunctionType, BT, VT} <:
       AbstractStructureFunction
    operator::OT
    distance::BT
    values::VT

    function StructureFunction(operator::OT, distance::BT, values::VT) where {OT, BT, VT}
        @assert length(distance) == length(values) "Distance and values must have the same length"
        FT = eltype(VT)
        return new{FT, OT, BT, VT}(operator, distance, values)
    end
end

"""
    StructureFunctionSumsAndCounts{FT, OT, BT, VT}

Intermediate result object containing raw sums and contribution counts.
Useful for aggregating measurements before final averaging.
"""
struct StructureFunctionSumsAndCounts{
    FT,
    OT <: SFT.AbstractStructureFunctionType,
    BT,
    VT,
} <: AbstractStructureFunction
    operator::OT
    distance::BT
    sums::VT
    counts::VT

    function StructureFunctionSumsAndCounts(
        operator::OT,
        distance::BT,
        sums::VT,
        counts::VT,
    ) where {OT, BT, VT}
        @assert length(distance) == length(sums) == length(counts) "Containers must have the same length"
        FT = eltype(VT)
        return new{FT, OT, BT, VT}(operator, distance, sums, counts)
    end
end

# ---------------------------------------------------------------------------
# Ergonomics & Base Extensions
# ---------------------------------------------------------------------------

import Base: show, length, +

Base.length(sf::AbstractStructureFunction) = length(sf.distance)

function Base.:+(sf1::StructureFunctionSumsAndCounts, sf2::StructureFunctionSumsAndCounts)
    @assert sf1.operator == sf2.operator "Cannot add results with different operators"
    @assert sf1.distance == sf2.distance "Cannot add results with different binning"
    return StructureFunctionSumsAndCounts(
        sf1.operator,
        sf1.distance,
        sf1.sums + sf2.sums,
        sf1.counts + sf2.counts,
    )
end

# Delegation to primary data container
Base.getindex(sf::StructureFunction, i...) = getindex(sf.values, i...)
Base.firstindex(sf::StructureFunction) = firstindex(sf.values)
Base.lastindex(sf::StructureFunction) = lastindex(sf.values)
Base.iterate(sf::StructureFunction, args...) = iterate(sf.values, args...)

# For SumsAndCounts, we don't necessarily want to treat it as a single array, 
# but getindex could perhaps return (sum, count) tuple? No, let's keep it explicit for now.
# Or better, just update the tests.

function Base.show(io::IO, sf::StructureFunction{FT, OT}) where {FT, OT}
    print(io, "StructureFunction{", FT, "}")
    print(io, "(operator=", sf.operator, ", points=", length(sf), ")")
end

function Base.show(io::IO, sf::StructureFunctionSumsAndCounts{FT, OT}) where {FT, OT}
    print(io, "StructureFunctionSumsAndCounts{", FT, "}")
    print(io, "(operator=", sf.operator, ", points=", length(sf), ")")
end

# Specialized getters
operator(sf::AbstractStructureFunction) = sf.operator
SFT.order(sf::AbstractStructureFunction) = SFT.order(sf.operator)

# Comparison & Conversion
import Base: isapprox, Float32, Float64

function Base.isapprox(sf1::StructureFunction, sf2::StructureFunction; kwargs...)
    return sf1.operator == sf2.operator &&
           sf1.distance == sf2.distance &&
           isapprox(sf1.values, sf2.values; kwargs...)
end

function Base.isapprox(sf::StructureFunction, vals::AbstractVector; kwargs...)
    return isapprox(sf.values, vals; kwargs...)
end
function Base.isapprox(vals::AbstractVector, sf::StructureFunction; kwargs...)
    return isapprox(sf.values, vals; kwargs...)
end

function Base.Float32(sf::StructureFunction)
    return StructureFunction(sf.operator, sf.distance, Float32.(sf.values))
end

function Base.Float64(sf::StructureFunction)
    return StructureFunction(sf.operator, sf.distance, Float64.(sf.values))
end

end # module
