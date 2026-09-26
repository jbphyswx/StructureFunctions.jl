using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Test: Test
using Distances: Distances as DI
using Random: Random
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA

Test.@testset "BinEdges Tests" begin
    for T in (Float64, Float32)
        # =====================================================================
        # 1. Generic BinEdges
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
        # 2. LinearBinEdges
        # =====================================================================
        lin_range = range(T(1.0), T(10.0), length=10)
        lin_be = SF.BinEdges(lin_range)

        Test.@test lin_be isa SF.LinearBinEdges
        Test.@test length(lin_be) == 10
        Test.@test lin_be[1] == T(1.0)
        Test.@test lin_be[10] == T(10.0)
        Test.@test lin_be[Int32(4)] === lin_be[4]
        lin_edges = collect(lin_be)
        for q in range(T(0.0), T(11.0), length=100)
            Test.@test searchsortedfirst(lin_be, q) == searchsortedfirst(lin_edges, q)
        end

        # =====================================================================
        # 3. LogBinEdges
        # =====================================================================
        log_be = SF.LogBinEdges(T(1.0), T(100.0), 15)
        log_range = range(log(T(1.0)), log(T(100.0)), length = 15)
        log_be_from_log = SF.LogBinEdges_from_log_edges(log_range)
        log_vec = collect(log_be)

        Test.@test length(log_be) == 15
        Test.@test log_be[1] == T(1.0)
        Test.@test log_be[15] == T(100.0)
        Test.@test log_vec ≈ exp.(log_range)
        Test.@test collect(log_be_from_log) ≈ exp.(log_range)
        Test.@test_throws ArgumentError SF.LogBinEdges(range(T(1.0), T(100.0); length = 15))

        queries = rand(T, 500) .* T(120.0)
        push!(queries, T(0.0), T(0.5), T(1.0), T(100.0), T(150.0))
        for e in log_vec
            push!(queries, e, prevfloat(e), nextfloat(e))
        end
        for q in queries
            Test.@test searchsortedfirst(log_be, q) == searchsortedfirst(log_vec, q)
        end

        # =====================================================================
        # 4. InfPaddedBinEdges
        # =====================================================================
        # Wrap generic vector
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

        # Wrapping a typed grid keeps its type and lookup.
        padded_lin_be = SF.InfPaddedBinEdges(lin_be)
        Test.@test padded_lin_be.edges === lin_be
        Test.@test length(padded_lin_be) == 12
        padded_log_be = SF.InfPaddedBinEdges(log_be)
        Test.@test padded_log_be.edges === log_be
        Test.@test length(padded_log_be) == 17

        # Test double-padding prevention
        double_padded = SF.InfPaddedBinEdges(T[typemin(T), 1.0, 2.0, typemax(T)])
        Test.@test length(double_padded) == 4
        Test.@test double_padded[1] == typemin(T)
        Test.@test double_padded[2] == T(1.0)
        Test.@test double_padded[3] == T(2.0)
        Test.@test double_padded[4] == typemax(T)
    end

    # A bin is a set of reals, so ±0 are one value: `searchsortedfirst` is one plus the number of edges
    # strictly below the query, and a NaN lies above every edge. Every lookup, every plan and every
    # squared plan equals that at each edge, its neighbouring floats, the infinities and both zeros.
    Test.@testset "exact edges" begin
        near(e) = (prevfloat(e, 2), prevfloat(e), e, nextfloat(e), nextfloat(e, 2))
        ref_first(edges, q) = count(e -> e < q, edges) + 1
        ref_last(edges, q) = count(e -> e <= q, edges)
        function queries(edges)
            T = eltype(edges)
            qs = T[q for e in edges if isfinite(e) for q in near(e)]
            return vcat(qs, T(-Inf), T(Inf), zero(T), -zero(T))
        end
        for T in (Float32, Float64)
            vecs = Any[
                SF.BinEdges(T[0, 0.07, 0.2, 0.21, 0.5, 0.9, 1.3]),
                SF.BinEdges(T[-Inf, 0, 0.5, 1, Inf]),
                SF.BinEdges(T[-1, -0.0, 1]),
                SF.BinEdges(T[0, 1e-6, 2e-6, 3e-6, 1, 1000]),
                SF.BinEdges(T[0.5, 2]),
            ]
            logs = Any[SF.LogBinEdges(T(0.1), T(10), 21), SF.LogBinEdges(T(1e-3), T(1e3), 61)]
            all_logs = Any[SF.LogBinEdges(T(1e-3), T(1e3), 7), logs...]
            cases = Any[
                SF.LinearBinEdges(T(0.1), T(1.7), 21),
                SF.LinearBinEdges(T(0), T(3), 31),
                SF.LinearBinEdges(T(-1), T(1), 21),
                SF.LinearBinEdges(T(-3.7), T(2.2), 60),
                SF.LinearBinEdges(T(1000), T(1100), 101),
                SF.LinearBinEdges(T(0), T(1), 2),
                SF.LogBinEdges(T(1e-3), T(1e3), 7),
                logs..., vecs...,
                map(SF.digitize_plan, logs)..., map(SF.digitize_plan, vecs)...,
                map(SF.LogTableBinEdges, all_logs)...,
            ]
            for inner in cases[[1, 7, 8, 10]]
                push!(cases, SF.InfPaddedBinEdges(inner), SF.digitize_plan(SF.InfPaddedBinEdges(inner)))
            end
            for inner in all_logs
                table = SF.LogTableBinEdges(inner)
                push!(cases, SF.InfPaddedBinEdges{T, typeof(table)}(table))
            end
            for b in cases
                edges = collect(b)
                Test.@test all(k -> edges[k] < edges[k + 1], 1:(length(edges) - 1))
                for q in queries(edges)
                    Test.@test searchsortedfirst(b, q) == ref_first(edges, q)
                    Test.@test searchsortedlast(b, q) == ref_last(edges, q)
                end
                # A padded grid counts a NaN in its last bin; every other grid puts it above the last edge.
                padded = b isa SF.InfPaddedBinEdges
                Test.@test searchsortedfirst(b, T(NaN)) == (padded ? length(edges) : length(edges) + 1)
                if T === Float32
                    for e in edges[isfinite.(edges)], q in near(Float64(e))
                        Test.@test searchsortedfirst(b, q) == ref_first(edges, q)
                    end
                end

                all(e -> !isfinite(e) || e >= 0, edges) || continue
                plan = SF.squared_digitize_plan(b)
                for e in edges
                    isfinite(e) && e > 0 || continue
                    for r2 in near(e * e)
                        r2 > 0 || continue
                        Test.@test SF.squared_digitize(plan, r2) == ref_first(edges, sqrt(r2)) - 1
                    end
                end
                Test.@test SF.squared_digitize(plan, T(Inf)) == ref_first(edges, T(Inf)) - 1
                Test.@test SF.squared_digitize(plan, T(NaN)) == (padded ? length(edges) - 1 : length(edges))
            end
        end
    end

    # The squared plans decide by thresholds `S_k`, the largest `s` with `sqrt(s) ≤ e_k`: every Float32
    # within 64 ulps of each threshold lands where `digitize(sqrt(s))` puts it.
    Test.@testset "squared thresholds, Float32 exhaustive" begin
        for b in (SF.LinearBinEdges(0.0f0, 3.0f0, 31), SF.LogBinEdges(1.0f-3, 1.0f3, 61),
                  SF.BinEdges(Float32[0.05, 0.1, 0.4, 0.45, 1.2]))
            edges = collect(b)
            plan = SF.squared_digitize_plan(b)
            bad = 0
            for e in edges
                s = e * e
                for k in -64:64
                    r2 = k < 0 ? prevfloat(s, -k) : nextfloat(s, k)
                    r2 > 0 || continue
                    bad += SF.squared_digitize(plan, r2) != searchsortedfirst(edges, sqrt(r2)) - 1
                end
            end
            Test.@test bad == 0
        end
    end

    # A lattice at spacing 0.1 puts many separations on edges at multiples of 0.1 and 0.125. Every point
    # route takes a pair's separation as `sqrt(HelperFunctions.norm2(dx))`, so every route bins it alike,
    # and the serial counts are that definition applied to each pair.
    Test.@testset "edge coincidences bin alike on every point route" begin
        x = reduce(hcat, [[0.1 * i, 0.1 * j] for i in 0:5 for j in 0:4])
        u = randn(Random.Xoshiro(3), 2, size(x, 2))
        sf = SFT.SecondOrderStructureFunctionType()
        backends = (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        for bins in (range(0.0, 0.5; length = 5), collect(0.0:0.1:0.5), SF.LinearBinEdges(0.0, 0.5, 6),
                     SF.LogBinEdges(0.1, 0.5, 5), SF.InfPaddedBinEdges(collect(0.1:0.1:0.4)))
            edges = collect(bins)
            want = zeros(Int, length(edges) - 1)
            for i in axes(x, 2), j in (i + 1):size(x, 2)
                dx = x[:, j] - x[:, i]
                b = count(e -> e < sqrt(SF.HelperFunctions.norm2(dx)), edges)
                1 <= b <= length(want) && (want[b] += 1)
            end
            one = [SFC.calculate_structure_function(sf, x, u, bins, SF.StructureFunctionSumsAndCounts; backend,
                       verbose = false, show_progress = false) for backend in backends]
            six = [SFC.calculate_structure_functions_single_pass(x, u, bins; backend).S2 for backend in backends
                   if !(bins isa SF.InfPaddedBinEdges)]
            Test.@test Int.(one[1].counts) == want
            for r in (one..., six...)
                Test.@test Int.(Array(r.counts)) == want
                Test.@test Array(r.sums) ≈ Array(one[1].sums)
            end
        end
    end

    Test.@testset "calculate_structure_function uses AbstractBinEdges in hot loop" begin
        Random.seed!(42)
        n = 40
        x = rand(2, n)
        u = randn(2, n)
        sft = SFT.L2SF
        for bins in (SF.LinearBinEdges(0.01, 2.0, 11), SF.LogBinEdges(0.01, 2.0, 11))
            typed = SFC.calculate_structure_function(
                sft, x, u, bins, SF.StructureFunctionSumsAndCounts;
                backend = CB.SerialBackend(), verbose = false, show_progress = false,
            )
            via_vector = SFC.calculate_structure_function(
                sft, x, u, collect(bins), SF.StructureFunctionSumsAndCounts;
                backend = CB.SerialBackend(), verbose = false, show_progress = false,
            )
            Test.@test via_vector.sums ≈ typed.sums
            Test.@test via_vector.counts == typed.counts
        end
    end

    Test.@testset "midpoints" begin
        # Every abscissa lies inside its own bin, for each way of expressing the same edges.
        for edges in (
            [0.0, 2.0, 6.0],
            SF.BinEdges([0.0, 2.0, 6.0]),
            SF.LinearBinEdges(range(0.0, 6.0; length = 4)),
            range(0.0, 6.0; length = 4),
            SF.LogBinEdges(1.0, 100.0, 3),
        )
            m = collect(SF.midpoints(edges))
            Test.@test length(m) == length(edges) - 1
            for k in eachindex(m)
                Test.@test edges[k] <= m[k] <= edges[k + 1]
            end
        end

        Test.@test SF.midpoints([0.0, 2.0, 6.0]) == [1.0, 4.0]
        Test.@test SF.midpoints(SF.BinEdges([0.0, 2.0, 6.0])) == [1.0, 4.0]
        Test.@test collect(SF.midpoints(SF.LinearBinEdges(range(0.0, 6.0; length = 4)))) ==
                   [1.0, 3.0, 5.0]
        Test.@test collect(SF.midpoints(range(0.0, 6.0; length = 4))) == [1.0, 3.0, 5.0]
        # Log edges are uniform on the log grid, so their abscissa is the geometric mean.
        Test.@test SF.midpoints(SF.LogBinEdges(1.0, 100.0, 3)) ≈ [sqrt(10.0), sqrt(1000.0)]

        out = zeros(2)
        Test.@test SF.midpoints!(out, [0.0, 2.0, 6.0]) == [1.0, 4.0]
        Test.@test SF.midpoints!(out, SF.LogBinEdges(1.0, 100.0, 3)) ≈
                   SF.midpoints(SF.LogBinEdges(1.0, 100.0, 3))
        Test.@test_throws DimensionMismatch SF.midpoints!(zeros(5), [0.0, 2.0, 6.0])
    end
end
