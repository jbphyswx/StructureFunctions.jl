using Test
using StructureFunctions
using StructureFunctions.Calculations
using StructureFunctions.StructureFunctionTypes
using StaticArrays
using LinearAlgebra

@testset "Phase 6: Deep Stability Checks" begin
    # 1. Forward-mode AD (Mock Dual)
    @testset "AD Sensitivity (Mock Dual)" begin
        struct MockDual{T} <: Real
            val::T
            der::T
        end
        Base.:+(a::MockDual, b::MockDual) = MockDual(a.val + b.val, a.der + b.der)
        Base.:-(a::MockDual, b::MockDual) = MockDual(a.val - b.val, a.der - b.der)
        Base.:*(a::MockDual, b::S) where {S<:Number} = MockDual(a.val * b, a.der * b)
        Base.:*(a::S, b::MockDual) where {S<:Number} = MockDual(a * b.val, a * b.der)
        Base.:*(a::MockDual, b::MockDual) = MockDual(a.val * b.val, a.val * b.der + a.der * b.val)
        Base.:/(a::MockDual, b::MockDual) = MockDual(a.val / b.val, (a.der * b.val - a.val * b.der) / (b.val^2))
        Base.:/(a::MockDual, b::S) where {S<:Number} = MockDual(a.val / b, a.der / b)
        Base.abs2(a::MockDual) = a * a
        Base.sqrt(a::MockDual) = MockDual(sqrt(a.val), 0.5 * a.der / sqrt(a.val))
        Base.promote_rule(::Type{MockDual{T}}, ::Type{S}) where {T, S<:Number} = MockDual{promote_type(T, S)}
        Base.convert(::Type{MockDual{T}}, x::T) where {T} = MockDual(x, zero(T))
        Base.convert(::Type{MockDual{T}}, x::S) where {T, S<:Number} = MockDual(convert(T, x), zero(T))
        Base.zero(::Type{MockDual{T}}) where {T} = MockDual(zero(T), zero(T))
        Base.isless(a::MockDual, b::MockDual) = a.val < b.val
        Base.float(a::MockDual) = a
        Base.abs(a::MockDual) = a.val >= 0 ? a : MockDual(-a.val, -a.der)
        
        # Self-conversion to avoid ambiguity
        Base.convert(::Type{MockDual{T}}, x::MockDual{T}) where {T} = x
        (::Type{T})(x::MockDual) where {T<:Number} = convert(T, x.val)

        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 0.0, 0.0]
        u_dual = [MockDual(1.0, 1.0), MockDual(2.0, 1.0), MockDual(3.0, 1.0)]
        v_dual = [MockDual(0.0, 0.0), MockDual(0.0, 0.0), MockDual(0.0, 0.0)]
        bins = SVector{1}([(0.1, 5.0)])
        
        sf_type = SecondOrderStructureFunction
        @test_nowarn calculate_structure_function(sf_type, (xs, ys), (u_dual, v_dual), bins; verbose=false, show_progress=false)
    end

    # 2. Degenerate/Collinear Points
    @testset "Degenerate Points (\$r=0\$)" begin
        # Duplicate points
        xs = [1.0, 1.0, 2.0]
        ys = [1.0, 1.0, 2.0]
        u = [1.0, 2.0, 3.0]
        v = [0.0, 0.0, 0.0]
        bins = SVector{1}([(0.1, 5.0)])
        
        sf_type = SecondOrderStructureFunction
        res, _ = calculate_structure_function(sf_type, (xs, ys), (u, v), bins; verbose=false, show_progress=false)
        @test !any(isnan.(res))
    end

    @testset "Collinear Points" begin
        # Points along a line
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 0.0, 0.0, 0.0]
        u = [1.0, 2.0, 3.0, 4.0]
        v = zeros(4)
        bins = SVector{2}([(0.1, 1.5), (1.5, 3.5)])
        
        sf_type = LongitudinalSecondOrderStructureFunction
        res, _ = calculate_structure_function(sf_type, (xs, ys), (u, v), bins; verbose=false, show_progress=false)
        @test length(res) == 2
        @test all(res .> 0)
    end
end
