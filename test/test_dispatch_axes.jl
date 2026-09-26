using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StaticArrays: StaticArrays as SA
using Random: Random

# A route picks its implementation from things that are not the numbers being computed: the
# coordinate width, the *type* of the bin edges, how many bins there are, and whether the
# positions are shared across slices. Every one of those is a branch, and a test that fixes one
# value of it exercises one side.
#
# Three defects reached the tree through exactly this blind spot — a device histogram that read
# two coordinates whatever the width, reachable only above a bin-count threshold; a shared-position
# batch that did the same, reachable only when the bins were spelled as `LinearBinEdges`; and a
# width axis the three backends answered three different ways, where serial computed a width the
# other two refused. All three were invisible to tests that used one shape, and the third was
# invisible to this file while its own width loop ran over `(2, 3)`.

const DA_DEV = CB.GPUBackend(KA.CPU())
const DA_SER = CB.SerialBackend()
const DA_OP = SFT.L2SFType()
const DA_RAW = SFO.StructureFunctionSumsAndCounts
const DA_N = 120
"""2 and 3 are the widths the device compiles ahead of time; 4 is the one that is not, which is
what separates a route that specializes a width from one that refuses it."""
const DA_WIDTHS = (2, 3, 4)

"""Counts exactly, sums to round-off."""
function da_agrees(got_s, got_c, ref_s, ref_c)
    collect(got_c) == collect(ref_c) || return false
    return isapprox(collect(got_s), collect(ref_s); rtol = 1e-10, atol = 1e-12)
end

"""The two spellings of the same edges: one carries its uniformity in its type, one does not."""
da_bin_spellings(lo, hi, n) = (
    ("typed", SF.LinearBinEdges(range(lo, hi; length = n + 1))),
    ("raw", collect(range(lo, hi; length = n + 1))),
)

Test.@testset "the device agrees across every axis a route dispatches on" begin
    Random.seed!(9100)

    Test.@testset "point 1D: width x bin spelling x bin count" begin
        for D in DA_WIDTHS, nb in (6, 300)
            x, u = rand(D, DA_N), randn(D, DA_N)
            for (spelling, bins) in da_bin_spellings(0.0, 1.0, nb)
                ref = SFC.calculate_structure_function(DA_OP, x, u, collect(bins), Float64, DA_RAW;
                    backend = DA_SER)
                got = SFC.calculate_structure_function(DA_OP, x, u, bins, Float64, DA_RAW;
                    backend = DA_DEV)
                Test.@test sum(ref.counts) > 0
                Test.@test (D, nb, spelling, da_agrees(got.sums, got.counts, ref.sums, ref.counts)) ==
                           (D, nb, spelling, true)
            end
        end
    end

    Test.@testset "joint 2D: width x bin count across the shared-memory cap" begin
        for D in DA_WIDTHS, nv in (5, 300)
            x, u = rand(D, DA_N), randn(D, DA_N)
            dbins = collect(range(0.0, 1.0; length = 7))
            vbins = collect(range(-3.0, 3.0; length = nv + 1))
            ref = SFC.calculate_structure_function(DA_OP, x, u, dbins, vbins;
                backend = DA_SER)
            got = SFC.calculate_structure_function(DA_OP, x, u, dbins, vbins;
                backend = DA_DEV)
            Test.@test sum(ref.counts) > 0
            Test.@test (D, nv, da_agrees(got.sums, got.counts, ref.sums, ref.counts)) ==
                       (D, nv, true)
        end
    end

    Test.@testset "the second axis x width: the angle is a property of the metric" begin
        dbins = collect(range(0.0, 1.0; length = 7))
        abins = collect(range(prevfloat(0.0), Float64(π); length = 5))
        for D in DA_WIDTHS
            x, u = rand(D, DA_N), randn(D, DA_N)
            ax = SFC.SeparationAngleAxis(SA.SVector(ntuple(d -> d == 1 ? 1.0 : 0.0, D)))
            ref = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                backend = DA_SER, second_axis = ax)
            Test.@test sum(ref.counts) > 0
            # without this control a backend that drops the keyword passes by accident
            val = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                backend = DA_SER)
            Test.@test collect(ref.counts) != collect(val.counts)
            for (name, be) in (("threaded", CB.ThreadedBackend()), ("device", DA_DEV))
                got = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                    backend = be, second_axis = ax)
                Test.@test (D, name, da_agrees(got.sums, got.counts, ref.sums, ref.counts)) ==
                           (D, name, true)
            end
        end
    end

    Test.@testset "single-pass 1D and 2D: width x bin spelling" begin
        for D in DA_WIDTHS
            x, u = rand(D, DA_N), randn(D, DA_N)
            vbins = collect(range(-3.0, 3.0; length = 6))
            for (spelling, bins) in da_bin_spellings(0.0, 1.0, 8)
                nb = length(collect(bins)) - 1
                rs = zeros(SFC.SINGLE_PASS_N, nb); rc = zeros(Int, SFC.SINGLE_PASS_N, nb)
                SFC.calculate_structure_functions_single_pass!(rs, rc, x, u, collect(bins);
                    backend = DA_SER)
                gs = zeros(SFC.SINGLE_PASS_N, nb); gc = zeros(Int, SFC.SINGLE_PASS_N, nb)
                SFC.calculate_structure_functions_single_pass!(gs, gc, x, u, bins;
                    backend = DA_DEV)
                Test.@test (D, spelling, :sp1d, da_agrees(gs, gc, rs, rc)) ==
                           (D, spelling, :sp1d, true)

                r2s = zeros(SFC.SINGLE_PASS_N, nb, 5); r2c = zeros(Int, SFC.SINGLE_PASS_N, nb, 5)
                SFC.calculate_structure_functions_single_pass_2d!(r2s, r2c, x, u, collect(bins),
                    vbins; backend = DA_SER)
                g2s = zeros(SFC.SINGLE_PASS_N, nb, 5); g2c = zeros(Int, SFC.SINGLE_PASS_N, nb, 5)
                SFC.calculate_structure_functions_single_pass_2d!(g2s, g2c, x, u, bins, vbins;
                    backend = DA_DEV)
                Test.@test (D, spelling, :sp2d, da_agrees(g2s, g2c, r2s, r2c)) ==
                           (D, spelling, :sp2d, true)
            end
        end
    end

    Test.@testset "slice batch: width x bin spelling x shared or varying positions" begin
        T = 3
        for D in DA_WIDTHS, shared in (true, false)
            u = randn(D, DA_N, T)
            x = shared ? rand(D, DA_N) : rand(D, DA_N, T)
            for (spelling, bins) in da_bin_spellings(0.0, 1.0, 8)
                nb = length(collect(bins)) - 1
                ref_s = zeros(nb, T); ref_c = zeros(Int, nb, T)
                for t in 1:T
                    xt = shared ? x : x[:, :, t]
                    r = SFC.calculate_structure_function(DA_OP, xt, u[:, :, t], collect(bins),
                        Float64, DA_RAW; backend = DA_SER)
                    ref_s[:, t] .= r.sums
                    ref_c[:, t] .= r.counts
                end
                g = SFC.calculate_structure_function(DA_OP, x, u, bins, DA_RAW; backend = DA_DEV)
                Test.@test sum(ref_c) > 0
                Test.@test (D, shared, spelling,
                            da_agrees(reshape(collect(g.sums), nb, T),
                                      reshape(collect(g.counts), nb, T), ref_s, ref_c)) ==
                           (D, shared, spelling, true)
            end
        end
    end

    Test.@testset "slice batch at a width no staged tile holds" begin
        # four Float64 tiles of 13 coordinates exceed every device's static shared memory
        D, T = 13, 3
        dbins = collect(range(0.0, 2.0; length = 7))
        vbins = collect(range(0.0, 8.0; length = 6))
        nb, nv, NI = length(dbins) - 1, length(vbins) - 1, SFC.SINGLE_PASS_N
        for shared in (true, false), weighted in (false, true)
            u = randn(D, DA_N, T)
            x = shared ? rand(D, DA_N) : rand(D, DA_N, T)
            kw = weighted ? (; weights = rand(DA_N)) : (;)
            CT = weighted ? Float64 : Int
            families = (
                (:batch1d, (nb, T), (s, c, be) -> SFC.calculate_structure_function_batch!(
                    s, c, DA_OP, x, u, dbins; backend = be, kw...)),
                (:joint, (nb, nv, T), (s, c, be) -> SFC.calculate_structure_function_2d_batch!(
                    s, c, DA_OP, x, u, dbins, vbins; backend = be, kw...)),
                (:sp1d, (NI, nb, T), (s, c, be) -> SFC.calculate_structure_functions_single_pass_batch!(
                    s, c, x, u, dbins; backend = be, kw...)),
                (:sp2d, (NI, nb, nv, T), (s, c, be) -> SFC.calculate_structure_functions_single_pass_2d_batch!(
                    s, c, x, u, dbins, vbins; backend = be, kw...)),
            )
            for (name, shape, run!) in families
                rs, rc = zeros(shape), zeros(CT, shape)
                run!(rs, rc, DA_SER)
                gs, gc = zeros(shape), zeros(CT, shape)
                run!(gs, gc, DA_DEV)
                agree = weighted ?
                    isapprox(gc, rc; rtol = 1e-10) && isapprox(gs, rs; rtol = 1e-10, atol = 1e-12) :
                    da_agrees(gs, gc, rs, rc)
                Test.@test sum(rc) > 0
                Test.@test (name, shared, weighted, agree) == (name, shared, weighted, true)
            end
        end
    end
end
