using StructureFunctions: StructureFunctions as SF
using Test: Test
using Distances: Distances as DI

Test.@testset "BinEdges Tests" begin
    for T in (Float64, Float32)
        # =====================================================================
        # 1. Fallback BinEdges
        # =====================================================================
        raw_vec = T[1.0, 2.5, 5.0, 10.0]
        be = SF.BinEdges(raw_vec)
        
        Test.@test length(be) == 4
        Test.@test size(be) == (4,)
        Test.@test be[1] == T(1.0)
        Test.@test be[4] == T(10.0)
        
        # searchsortedfirst correctness
        Test.@test searchsortedfirst(be, T(0.5)) == 1
        Test.@test searchsortedfirst(be, T(1.0)) == 1
        Test.@test searchsortedfirst(be, T(2.0)) == 2
        Test.@test searchsortedfirst(be, T(2.5)) == 2
        Test.@test searchsortedfirst(be, T(8.0)) == 4
        Test.@test searchsortedfirst(be, T(10.0)) == 4
        Test.@test searchsortedfirst(be, T(12.0)) == 5
        
        # =====================================================================
        # 2. LinearBinEdges (FMA Linear Spacing)
        # =====================================================================
        lin_range = range(T(1.0), T(10.0), length=10)
        lin_be = SF.BinEdges(lin_range)
        
        Test.@test lin_be isa SF.LinearBinEdges
        Test.@test length(lin_be) == 10
        Test.@test lin_be[1] == T(1.0)
        Test.@test lin_be[10] == T(10.0)
        
        # Query correctness across a set of points
        for q in range(T(0.0), T(11.0), length=100)
            ref = searchsortedfirst(lin_range, q)
            Test.@test searchsortedfirst(lin_be, q) == ref
        end
        
        # =====================================================================
        # 3. LogBinEdges (Exponent LUT Hybrid Spacing)
        # =====================================================================
        log_range = range(log(T(1.0)), log(T(100.0)), length=15)
        log_vec = exp.(log_range)
        log_be = SF.LogBinEdges(log_vec)
        
        Test.@test log_be isa SF.LogBinEdges
        Test.@test length(log_be) == 15
        Test.@test log_be[1] == log_vec[1]
        Test.@test log_be[15] == log_vec[15]
        
        # Verify correctness with random queries in the range
        queries = rand(T, 500) .* T(120.0)
        # Include boundary queries
        push!(queries, T(0.0), T(0.5), T(1.0), T(100.0), T(150.0))
        for i in 1:length(log_vec)
            push!(queries, log_vec[i])
            push!(queries, log_vec[i] - eps(log_vec[i]))
            push!(queries, log_vec[i] + eps(log_vec[i]))
        end
        
        for q in queries
            ref = searchsortedfirst(log_vec, q)
            Test.@test searchsortedfirst(log_be, q) == ref
        end

        # =====================================================================
        # 4. InfPaddedBinEdges
        # =====================================================================
        # Wrap fallback vector
        padded_be = SF.InfPaddedBinEdges(be)
        Test.@test length(padded_be) == 6
        Test.@test padded_be[1] == typemin(T)
        Test.@test padded_be[2] == T(1.0)
        Test.@test padded_be[5] == T(10.0)
        Test.@test padded_be[6] == typemax(T)
        
        # Test boundaries
        Test.@test searchsortedfirst(padded_be, typemin(T)) == 1
        Test.@test searchsortedfirst(padded_be, T(0.5)) == 2 # falls in (-Inf, 1.0]
        Test.@test searchsortedfirst(padded_be, T(1.0)) == 2
        Test.@test searchsortedfirst(padded_be, T(2.0)) == 3 # falls in (1.0, 2.5]
        Test.@test searchsortedfirst(padded_be, T(10.0)) == 5
        Test.@test searchsortedfirst(padded_be, T(15.0)) == 6
        Test.@test searchsortedfirst(padded_be, typemax(T)) == 6

        # Test searchsortedlast
        Test.@test searchsortedlast(padded_be, T(0.5)) == 1
        Test.@test searchsortedlast(padded_be, T(1.0)) == 2
        Test.@test searchsortedlast(padded_be, T(1.5)) == 2
        Test.@test searchsortedlast(padded_be, T(10.0)) == 5
        Test.@test searchsortedlast(padded_be, T(15.0)) == 5
        Test.@test searchsortedlast(padded_be, typemax(T)) == 6

        # Wrap LinearBinEdges
        padded_lin_be = SF.InfPaddedBinEdges(lin_be)
        Test.@test length(padded_lin_be) == 12
        Test.@test padded_lin_be[1] == typemin(T)
        Test.@test padded_lin_be[12] == typemax(T)
        for q in range(T(-1.0), T(12.0), length=100)
            # Find index conceptually in padded vector
            ref_idx = searchsortedfirst(lin_range, q) + 1
            if q <= typemin(T)
                ref_idx = 1
            elseif q > last(lin_range)
                ref_idx = length(lin_range) + 2
            end
            Test.@test searchsortedfirst(padded_lin_be, q) == ref_idx
        end

        # Wrap LogBinEdges
        padded_log_be = SF.InfPaddedBinEdges(log_be)
        Test.@test length(padded_log_be) == 17
        Test.@test padded_log_be[1] == typemin(T)
        Test.@test padded_log_be[17] == typemax(T)
        for q in queries
            ref_idx = searchsortedfirst(log_vec, q) + 1
            if q <= typemin(T)
                ref_idx = 1
            elseif q > last(log_vec)
                ref_idx = length(log_vec) + 2
            end
            Test.@test searchsortedfirst(padded_log_be, q) == ref_idx
        end
        
        # Test double-padding prevention
        double_padded = SF.InfPaddedBinEdges(T[typemin(T), 1.0, 2.0, typemax(T)])
        Test.@test length(double_padded) == 4
        Test.@test double_padded[1] == typemin(T)
        Test.@test double_padded[2] == T(1.0)
        Test.@test double_padded[3] == T(2.0)
        Test.@test double_padded[4] == typemax(T)
    end
end
