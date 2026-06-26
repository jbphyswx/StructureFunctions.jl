module StructureFunctionObjects

using ..StructureFunctionTypes: StructureFunctionTypes as SFT

export AbstractStructureFunction,
    StructureFunction,
    StructureFunctionSumsAndCounts,
    StructureFunction2DSumsAndCounts,
    StructureFunctionTensor,
    StructureFunctionTensorSumsAndCounts,
    HelmholtzDecomposition2D,
    marginalize

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
        n_values = values isa AbstractArray && ndims(values) > 1 ? size(values, 1) : length(values)
        (length(distance) == n_values + 1) || throw(DimensionMismatch("Flat distance edges must have length one greater than the leading value-bin axis (got edges=$(length(distance)), value bins=$n_values)"))
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
        n_sums = sums isa AbstractArray && ndims(sums) > 1 ? size(sums, 1) : length(sums)
        n_counts = counts isa AbstractArray && ndims(counts) > 1 ? size(counts, 1) : length(counts)
        ((length(distance) == n_sums + 1) && (n_sums == n_counts) && (size(sums) == size(counts))) || throw(DimensionMismatch("Flat distance edges must satisfy length(distance) == size(sums,1) + 1 and sums/counts must match shape (got edges=$(length(distance)), sums=$(size(sums)), counts=$(size(counts)))"))
        FT = eltype(sums)
        return new{FT, OT, BT, VT, CT}(operator, distance, sums, counts)
    end
end

"""
    StructureFunction2DSumsAndCounts{FT, OT, BT, VT, MT}

2D Joint-Probability intermediate result container containing raw sums and counts matrices.
Useful for analyzing the PDF of structure function values across separation distance bins.

- `operator`: The specific operator used (e.g., `L2SF`).
- `distance_bins`: 1D container of distance bin edges.
- `value_bins`: 1D container of structure function value bin edges.
- `sums`: 2D matrix of accumulated exact values of shape (N_distance_bins, N_value_bins).
- `counts`: 2D matrix of accumulated contribution counts of shape (N_distance_bins, N_value_bins).
"""
struct StructureFunction2DSumsAndCounts{
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

    function StructureFunction2DSumsAndCounts(
        operator::OT,
        distance_bins::BT,
        value_bins::VT,
        sums::MT,
        counts::CT,
    ) where {OT, BT, VT, MT, CT}
        (size(sums) == size(counts)) || throw(DimensionMismatch("Sums and counts matrices must have identical shape (got sums: $(size(sums)), counts: $(size(counts)))"))
        (size(sums, 1) == length(distance_bins) - 1) || throw(DimensionMismatch("distance_bins must have length size(sums,1)+1 (got $(length(distance_bins)) edges, $(size(sums,1)) distance bins)"))
        (size(sums, 2) == length(value_bins) - 1) || throw(DimensionMismatch("value_bins must have length size(sums,2)+1 (got $(length(value_bins)) edges, $(size(sums,2)) value bins)"))
        FT = eltype(sums)
        return new{FT, OT, BT, VT, MT, CT}(operator, distance_bins, value_bins, sums, counts)
    end
end

"""
    StructureFunctionTensorSumsAndCounts(order, distance_bins, sums, counts)

Raw binned tensor structure-function result. For tensor order `P` and spatial
dimension `D`, `sums` has leading axes `(D, D, ..., D, n_bins, auxiliary...)`
with `P` repeated component axes. `counts` has shape `(n_bins, auxiliary...)`.
"""
struct StructureFunctionTensorSumsAndCounts{P, FT, BT, VT, CT} <: AbstractStructureFunction
    order::Val{P}
    distance_bins::BT
    sums::VT
    counts::CT

    function StructureFunctionTensorSumsAndCounts(
        order::Val{P},
        distance_bins::BT,
        sums::VT,
        counts::CT,
    ) where {P, BT, VT, CT}
        P >= 1 || throw(ArgumentError("tensor order must be positive"))
        ndims(sums) >= P + 1 ||
            throw(DimensionMismatch("tensor sums must have at least P component axes plus one distance-bin axis"))
        n_bins = size(sums, P + 1)
        length(distance_bins) == n_bins + 1 ||
            throw(DimensionMismatch("distance_bins must have length size(sums,$(P + 1))+1"))
        size(counts, 1) == n_bins ||
            throw(DimensionMismatch("counts first axis must match tensor distance-bin axis"))
        size(counts)[2:end] == size(sums)[(P + 2):end] ||
            throw(DimensionMismatch("counts auxiliary axes must match tensor sums auxiliary axes"))
        FT = eltype(sums)
        return new{P, FT, BT, VT, CT}(order, distance_bins, sums, counts)
    end
end

"""
    StructureFunctionTensor(order, distance_bins, values)

Averaged binned tensor structure function — the mean tensor ``D_{i…}(r)`` per distance bin
(the `sums ./ counts` reduction of a [`StructureFunctionTensorSumsAndCounts`](@ref), with the
empty-bin guard `count == 0 → NaN`). For tensor order `P` and spatial dimension `D`, `values`
has leading axes `(D, D, ..., D, n_bins, auxiliary...)` with `P` repeated component axes. This
is the default result of `calculate_structure_function_tensor`; pass
`output_type = StructureFunctionTensorSumsAndCounts` for the raw accumulator.
"""
struct StructureFunctionTensor{P, FT, BT, VT} <: AbstractStructureFunction
    order::Val{P}
    distance_bins::BT
    values::VT

    function StructureFunctionTensor(order::Val{P}, distance_bins::BT, values::VT) where {P, BT, VT}
        P >= 1 || throw(ArgumentError("tensor order must be positive"))
        ndims(values) >= P + 1 ||
            throw(DimensionMismatch("tensor values must have at least P component axes plus one distance-bin axis"))
        n_bins = size(values, P + 1)
        length(distance_bins) == n_bins + 1 ||
            throw(DimensionMismatch("distance_bins must have length size(values,$(P + 1))+1"))
        FT = eltype(values)
        return new{P, FT, BT, VT}(order, distance_bins, values)
    end
end

"""
    HelmholtzDecomposition2D(distance_bins, rotational_sums, rotational_counts,
                             divergent_sums, divergent_counts,
                             longitudinal_values, transverse_values)

Result of the 2D isotropic Helmholtz decomposition computed from binned
longitudinal/transverse second-order structure functions. Rotational and
divergent components are derived quantities, not pairwise operators.
"""
struct HelmholtzDecomposition2D{FT, BT, VS, VC, VV} <: AbstractStructureFunction
    operator::SFT.HelmholtzDecomposition2DType
    distance_bins::BT
    rotational_sums::VS
    rotational_counts::VC
    divergent_sums::VS
    divergent_counts::VC
    longitudinal_values::VV
    transverse_values::VV

    function HelmholtzDecomposition2D(
        distance_bins::BT,
        rotational_sums::VS,
        rotational_counts::VC,
        divergent_sums::VS,
        divergent_counts::VC,
        longitudinal_values::VV,
        transverse_values::VV,
    ) where {BT, VS, VC, VV}
        n_bins = length(distance_bins) - 1
        length(rotational_sums) == n_bins ||
            throw(DimensionMismatch("rotational_sums must have length $n_bins"))
        length(rotational_counts) == n_bins ||
            throw(DimensionMismatch("rotational_counts must have length $n_bins"))
        length(divergent_sums) == n_bins ||
            throw(DimensionMismatch("divergent_sums must have length $n_bins"))
        length(divergent_counts) == n_bins ||
            throw(DimensionMismatch("divergent_counts must have length $n_bins"))
        length(longitudinal_values) == n_bins ||
            throw(DimensionMismatch("longitudinal_values must have length $n_bins"))
        length(transverse_values) == n_bins ||
            throw(DimensionMismatch("transverse_values must have length $n_bins"))
        FT = eltype(rotational_sums)
        return new{FT, BT, VS, VC, VV}(
            SFT.HelmholtzDecomposition2DOperator,
            distance_bins,
            rotational_sums,
            rotational_counts,
            divergent_sums,
            divergent_counts,
            longitudinal_values,
            transverse_values,
        )
    end
end

# ---------------------------------------------------------------------------
# Ergonomics & Base Extensions
# ---------------------------------------------------------------------------

import Base: show, length, +

Base.length(sf::StructureFunction) = length(sf.values)
Base.length(sf::StructureFunctionSumsAndCounts) = length(sf.sums)
Base.length(sf::StructureFunction2DSumsAndCounts) = length(sf.distance_bins) - 1
Base.length(sf::StructureFunctionTensorSumsAndCounts) = length(sf.distance_bins) - 1
Base.length(sf::StructureFunctionTensor) = length(sf.distance_bins) - 1
Base.length(sf::HelmholtzDecomposition2D) = length(sf.distance_bins) - 1

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

function Base.:+(sf1::StructureFunctionTensorSumsAndCounts{P}, sf2::StructureFunctionTensorSumsAndCounts{P}) where {P}
    (sf1.distance_bins == sf2.distance_bins) ||
        throw(ArgumentError("Cannot add tensor results with different distance binning"))
    return StructureFunctionTensorSumsAndCounts(
        sf1.order,
        sf1.distance_bins,
        sf1.sums + sf2.sums,
        sf1.counts + sf2.counts,
    )
end

function Base.:+(sf1::StructureFunction2DSumsAndCounts, sf2::StructureFunction2DSumsAndCounts)
    (sf1.operator == sf2.operator) || throw(ArgumentError("Cannot add results with different operators: got $(sf1.operator) and $(sf2.operator)"))
    (sf1.distance_bins == sf2.distance_bins) || throw(ArgumentError("Cannot add results with different distance binning"))
    (sf1.value_bins == sf2.value_bins) || throw(ArgumentError("Cannot add results with different value binning"))
    return StructureFunction2DSumsAndCounts(
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

function Base.show(io::IO, sf::StructureFunction2DSumsAndCounts{FT, OT}) where {FT, OT}
    print(io, "StructureFunction2DSumsAndCounts{", FT, "}")
    print(io, "(operator=", sf.operator, ", distance_bins=", length(sf.distance_bins), ", value_bins=", length(sf.value_bins), ")")
end

function Base.show(io::IO, sf::StructureFunctionTensorSumsAndCounts{P, FT}) where {P, FT}
    print(io, "StructureFunctionTensorSumsAndCounts{", P, ", ", FT, "}")
    print(io, "(distance_bins=", length(sf.distance_bins), ", size=", size(sf.sums), ")")
end

function Base.show(io::IO, sf::StructureFunctionTensor{P, FT}) where {P, FT}
    print(io, "StructureFunctionTensor{", P, ", ", FT, "}")
    print(io, "(distance_bins=", length(sf.distance_bins), ", size=", size(sf.values), ")")
end

function Base.show(io::IO, sf::HelmholtzDecomposition2D{FT}) where {FT}
    print(io, "HelmholtzDecomposition2D{", FT, "}")
    print(io, "(distance_bins=", length(sf.distance_bins), ")")
end

# Specialized getters
operator(sf::AbstractStructureFunction) = sf.operator
SFT.order(sf::AbstractStructureFunction) = SFT.order(sf.operator)
SFT.order(sf::StructureFunctionTensorSumsAndCounts{P}) where {P} = P
SFT.order(sf::StructureFunctionTensor{P}) where {P} = P

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
    marginalize(sf2d::StructureFunction2DSumsAndCounts)

Sum ``sums`` and ``counts`` over the value-bin axis to produce a 1D
``StructureFunctionSumsAndCounts`` (mass-conserving reduction of a 2D joint histogram).
"""
function marginalize(sf2d::StructureFunction2DSumsAndCounts)
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
