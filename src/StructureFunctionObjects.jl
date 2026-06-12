module StructureFunctionObjects

using ..StructureFunctionTypes: StructureFunctionTypes as SFT

export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts, StructureFunction2D, marginalize

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
        (length(distance) == length(values)) || throw(DimensionMismatch("Distance and values must have the same length (got $(length(distance)) and $(length(values)))"))
        FT = eltype(VT)
        return new{FT, OT, BT, VT}(operator, distance, values)
    end
end

"""
    StructureFunctionSumsAndCounts{FT, OT, BT, VT}

Intermediate result object containing raw sums and per-bin pair counts (integer).
Useful for aggregating measurements before final averaging.
"""
struct StructureFunctionSumsAndCounts{
    FT,
    OT <: SFT.AbstractStructureFunctionType,
    BT,
    VT,
    CT,
} <: AbstractStructureFunction
    operator::OT
    distance::BT
    sums::VT
    counts::CT

    function StructureFunctionSumsAndCounts(
        operator::OT,
        distance::BT,
        sums::VT,
        counts::CT,
    ) where {OT, BT, VT, CT}
        ((length(distance) == length(sums)) && (length(sums) == length(counts))) || throw(DimensionMismatch("Containers must have the same length (got distance: $(length(distance)), sums: $(length(sums)), counts: $(length(counts)))"))
        FT = eltype(sums)
        return new{FT, OT, BT, VT, CT}(operator, distance, sums, counts)
    end
end

"""
    StructureFunction2D{FT, OT, BT, VT, MT}

2D Joint-Probability intermediate result container containing raw sums and counts matrices.
Useful for analyzing the PDF of structure function values across separation distance bins.

- `operator`: The specific operator used (e.g., `L2SF`).
- `distance_bins`: 1D container of distance bin edges.
- `value_bins`: 1D container of structure function value bin edges.
- `sums`: 2D matrix of accumulated exact values of shape (N_distance_bins, N_value_bins).
- `counts`: 2D matrix of accumulated contribution counts of shape (N_distance_bins, N_value_bins).
"""
struct StructureFunction2D{
    FT,
    OT <: SFT.AbstractStructureFunctionType,
    BT,
    VT,
    MT,
    CT,
} <: AbstractStructureFunction
    operator::OT
    distance_bins::BT
    value_bins::VT
    sums::MT
    counts::CT

    function StructureFunction2D(
        operator::OT,
        distance_bins::BT,
        value_bins::VT,
        sums::MT,
        counts::CT,
    ) where {OT, BT, VT, MT, CT}
        (size(sums) == size(counts)) || throw(DimensionMismatch("Sums and counts matrices must have identical shape (got sums: $(size(sums)), counts: $(size(counts)))"))
        FT = eltype(sums)
        return new{FT, OT, BT, VT, MT, CT}(operator, distance_bins, value_bins, sums, counts)
    end
end

# ---------------------------------------------------------------------------
# Ergonomics & Base Extensions
# ---------------------------------------------------------------------------

import Base: show, length, +

Base.length(sf::StructureFunction) = length(sf.distance)
Base.length(sf::StructureFunctionSumsAndCounts) = length(sf.distance)
Base.length(sf::StructureFunction2D) = length(sf.distance_bins)

function Base.:+(sf1::StructureFunctionSumsAndCounts, sf2::StructureFunctionSumsAndCounts)
    (sf1.operator == sf2.operator) || throw(ArgumentError("Cannot add results with different operators: got $(sf1.operator) and $(sf2.operator)"))
    (sf1.distance == sf2.distance) || throw(ArgumentError("Cannot add results with different binning"))
    return StructureFunctionSumsAndCounts(
        sf1.operator,
        sf1.distance,
        sf1.sums + sf2.sums,
        sf1.counts + sf2.counts,
    )
end

function Base.:+(sf1::StructureFunction2D, sf2::StructureFunction2D)
    (sf1.operator == sf2.operator) || throw(ArgumentError("Cannot add results with different operators: got $(sf1.operator) and $(sf2.operator)"))
    (sf1.distance_bins == sf2.distance_bins) || throw(ArgumentError("Cannot add results with different distance binning"))
    (sf1.value_bins == sf2.value_bins) || throw(ArgumentError("Cannot add results with different value binning"))
    return StructureFunction2D(
        sf1.operator,
        sf1.distance_bins,
        sf1.value_bins,
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

function Base.show(io::IO, sf::StructureFunction2D{FT, OT}) where {FT, OT}
    print(io, "StructureFunction2D{", FT, "}")
    print(io, "(operator=", sf.operator, ", distance_bins=", length(sf.distance_bins), ", value_bins=", length(sf.value_bins), ")")
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

"""
    marginalize(sf2d::StructureFunction2D)

Sum ``sums`` and ``counts`` over the value-bin axis to produce a 1D
``StructureFunctionSumsAndCounts`` (mass-conserving reduction of a 2D joint histogram).
"""
function marginalize(sf2d::StructureFunction2D)
    sums_1d = vec(sum(sf2d.sums, dims = 2))
    counts_1d = vec(sum(sf2d.counts, dims = 2))
    return StructureFunctionSumsAndCounts(
        sf2d.operator,
        sf2d.distance_bins,
        sums_1d,
        counts_1d,
    )
end

end # module
