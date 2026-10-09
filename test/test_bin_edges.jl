using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Test: Test
using Random: Random

# (backend, single pass): each serial pair kernel that forms a separation; test_device.jl runs these on the device.
const BE_COINCIDENCE_ROUTES = ((CB.SerialBackend(), false), (CB.SerialBackend(), true))

Test.@testset "BinEdges Tests" begin
    # Uniform edges are their grid at any integer index; padding adds the infinite ends once.
    Test.@testset "typed and padded edges hold the edges they are built from" begin
        for T in (Float64, Float32)
            lin_range = range(T(1.0), T(10.0), length = 10)
            lin_be = SF.BinEdges(lin_range)
            Test.@test collect(lin_be) ≈ lin_range
            Test.@test lin_be[Int32(4)] === lin_be[4]
            Test.@test all(b -> collect(SF.InfPaddedBinEdges(b)) == [typemin(T); collect(b); typemax(T)],
                           (SF.BinEdges(T[1.0, 2.5, 5.0, 10.0]), lin_be, SF.LogBinEdges(T(1.0), T(100.0), 15)))
            Test.@test collect(SF.InfPaddedBinEdges(T[typemin(T), 1, 2, typemax(T)])) == T[typemin(T), 1, 2, typemax(T)]
        end
    end

    # A bin holds one plus the edges strictly below a query (±0 one value, NaN above all), at each edge's neighbours.
    Test.@testset "exact edges" begin
        near(e) = (prevfloat(e, 2), prevfloat(e), e, nextfloat(e), nextfloat(e, 2))
        ref_first(edges, q) = count(e -> e < q, edges) + 1
        ref_last(edges, q) = count(e -> e <= q, edges)
        function queries(edges)
            T = eltype(edges)
            qs = T[q for e in edges if isfinite(e) for q in near(e)]
            return vcat(qs, T(-Inf), T(Inf), zero(T), -zero(T))
        end
        misses = (; increasing = Any[], first = Any[], last = Any[], nan = Any[], wide_query = Any[], squared = Any[],
                  in_range = Any[], select = Any[], squared_inf = Any[], squared_nan = Any[])
        for T in (Float32, Float64)
            lins = Any[SF.LinearBinEdges(T(-3.7), T(2.2), 60), SF.LinearBinEdges(T(1000), T(1100), 101),
                       SF.LinearBinEdges(T(0), T(1), 2)]
            logb = SF.LogBinEdges(T(1e-3), T(1e3), 61)
            vecs = Any[SF.BinEdges(T[-Inf, 0, 0.5, 1, Inf]), SF.BinEdges(T[-1, -0.0, 1]),
                       SF.BinEdges(T[0, 1e-6, 2e-6, 3e-6, 1, 1000])]
            table = SF.LogTableBinEdges(logb)
            cases = Any[lins..., logb, vecs..., SF.digitize_plan(logb), map(SF.digitize_plan, vecs)..., table]
            for inner in (lins[3], logb, vecs[3])
                push!(cases, SF.InfPaddedBinEdges(inner), SF.digitize_plan(SF.InfPaddedBinEdges(inner)))
            end
            push!(cases, SF.InfPaddedBinEdges{T, typeof(table)}(table))
            for (i, b) in enumerate(cases)
                case = (T, i, typeof(b))
                edges = collect(b)
                all(k -> edges[k] < edges[k + 1], 1:(length(edges) - 1)) || push!(misses.increasing, case)
                for q in queries(edges)
                    searchsortedfirst(b, q) == ref_first(edges, q) || push!(misses.first, (case, q))
                    searchsortedlast(b, q) == ref_last(edges, q) || push!(misses.last, (case, q))
                end
                padded = b isa SF.InfPaddedBinEdges
                searchsortedfirst(b, T(NaN)) == (padded ? length(edges) : length(edges) + 1) || push!(misses.nan, case)
                if T === Float32
                    for e in edges[isfinite.(edges)], q in near(Float64(e))
                        searchsortedfirst(b, q) == ref_first(edges, q) || push!(misses.wide_query, (case, q))
                    end
                end

                all(e -> !isfinite(e) || e >= 0, edges) || continue
                plan = SF.squared_digitize_plan(b)
                for e in edges
                    isfinite(e) && e > 0 || continue
                    for r2 in near(e * e)
                        r2 > 0 || continue
                        SF.squared_digitize(plan, r2) == ref_first(edges, sqrt(r2)) - 1 || push!(misses.squared, (case, r2))
                        key, idx = SF.digitize_key(plan, r2), SF.squared_approx_index(plan, r2)
                        bin = SF.squared_bin(plan, key, idx)
                        SF.squared_in_range(plan, key) == (1 <= bin <= SF.n_histogram_bins(plan)) ||
                            push!(misses.in_range, (case, r2))
                        SF.squared_bin_select(plan, key, idx) == bin || push!(misses.select, (case, r2))
                    end
                end
                SF.squared_digitize(plan, T(Inf)) == ref_first(edges, T(Inf)) - 1 || push!(misses.squared_inf, case)
                SF.squared_digitize(plan, T(NaN)) == (padded ? length(edges) - 1 : length(edges)) ||
                    push!(misses.squared_nan, case)
            end
        end
        for (property, missed) in pairs(misses)
            Test.@test (property, missed) == (property, Any[])
        end
    end

    # A lattice at spacing 0.1 puts many separations on edges; each kernel bins them as `sqrt(norm2(dx))` against the edges.
    Test.@testset "edge coincidences bin alike on every point kernel" begin
        x = reduce(hcat, [[0.1 * i, 0.1 * j] for i in 0:5 for j in 0:4])
        u = randn(Random.Xoshiro(3), 2, size(x, 2))
        sf = SFT.SecondOrderStructureFunctionType()
        bins = collect(0.0:0.1:0.5)
        for (backend, single_pass) in BE_COINCIDENCE_ROUTES
            edges = collect(bins)
            want = zeros(Int, length(edges) - 1)
            want_sums = zeros(length(edges) - 1)
            for i in axes(x, 2), j in (i + 1):size(x, 2)
                dx = x[:, j] - x[:, i]
                b = count(e -> e < sqrt(SF.HelperFunctions.norm2(dx)), edges)
                if 1 <= b <= length(want)
                    want[b] += 1
                    want_sums[b] += sum(abs2, u[:, j] - u[:, i])
                end
            end
            r = single_pass ? SFC.calculate_structure_functions_single_pass(x, u, bins; backend).S2 :
                SFC.calculate_structure_function(sf, x, u, bins, SF.StructureFunctionSumsAndCounts; backend)
            Test.@test Int.(Array(r.counts)) == want
            Test.@test Array(r.sums) ≈ want_sums
        end
    end

    Test.@testset "midpoints" begin
        Test.@test SF.midpoints([0.0, 2.0, 6.0]) == [1.0, 4.0]
        Test.@test collect(SF.midpoints(range(0.0, 6.0; length = 4))) == [1.0, 3.0, 5.0]
        Test.@test SF.midpoints!(zeros(2), SF.LogBinEdges(1.0, 100.0, 3)) ≈ [sqrt(10.0), sqrt(1000.0)]
        Test.@test_throws DimensionMismatch SF.midpoints!(zeros(5), [0.0, 2.0, 6.0])
    end
end
