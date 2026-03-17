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
struct StructureFunction{FT, OT <: SFT.AbstractStructureFunctionType, BT, VT} <: AbstractStructureFunction
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
struct StructureFunctionSumsAndCounts{FT, OT <: SFT.AbstractStructureFunctionType, BT, VT} <: AbstractStructureFunction
    operator::OT
    distance::BT
    sums::VT
    counts::VT

    function StructureFunctionSumsAndCounts(operator::OT, distance::BT, sums::VT, counts::VT) where {OT, BT, VT}
        @assert length(distance) == length(sums) == length(counts) "Containers must have the same length"
        FT = eltype(VT)
        return new{FT, OT, BT, VT}(operator, distance, sums, counts)
    end
end

# ---------------------------------------------------------------------------
# Ergonomics & Base Extensions
# ---------------------------------------------------------------------------

import Base: show, length

Base.length(sf::AbstractStructureFunction) = length(sf.distance)

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

end # module
