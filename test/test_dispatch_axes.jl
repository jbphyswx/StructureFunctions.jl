using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StaticArrays: StaticArrays as SA
using Random: Random

const DA_DEV = CB.GPUBackend(KA.CPU())
const DA_SER = CB.SerialBackend()
const DA_OP = SFT.L2SFType()
const DA_RAW = SFO.StructureFunctionSumsAndCounts
const DA_N = 120

"""Counts exactly, sums to round-off."""
function da_agrees(got_s, got_c, ref_s, ref_c)
    collect(got_c) == collect(ref_c) || return false
    return isapprox(collect(got_s), collect(ref_s); rtol = 1e-10, atol = 1e-12)
end

"""`n` equal bins over `[lo, hi]`: `LinearBinEdges` for `:typed`, a plain vector for `:raw`."""
da_bins(spelling, lo, hi, n) = spelling === :typed ? SF.LinearBinEdges(range(lo, hi; length = n + 1)) :
                               collect(range(lo, hi; length = n + 1))

"""Batch family `name`: its output shape, its batch call on a backend, and its serial call on one slice `(xt, ut)`."""
function da_batch_family(name, x, u, dbins, vbins, T, kw)
    nb, nv, NI = length(dbins) - 1, length(vbins) - 1, SFC.SINGLE_PASS_N
    name === :batch1d && return ((nb, T),
        ((s, c, be) -> SFC.calculate_structure_function_batch!(s, c, DA_OP, x, u, dbins; backend = be, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_function!(s, c, DA_OP, xt, ut, dbins; backend = DA_SER, kw...)))
    name === :joint && return ((nb, nv, T),
        ((s, c, be) -> SFC.calculate_structure_function_2d_batch!(s, c, DA_OP, x, u, dbins, vbins; backend = be, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_function!(s, c, DA_OP, xt, ut, dbins, vbins; backend = DA_SER,
                                                             kw...)))
    name === :sp1d && return ((NI, nb, T),
        ((s, c, be) -> SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, dbins; backend = be, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_functions_single_pass!(s, c, xt, ut, dbins; backend = DA_SER,
                                                                          kw...)))
    name === :sp2d && return ((NI, nb, nv, T),
        ((s, c, be) -> SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, dbins, vbins; backend = be,
                                                                               kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_functions_single_pass_2d!(s, c, xt, ut, dbins, vbins;
                                                                             backend = DA_SER, kw...)))
    error("unknown batch family $name")
end

const DA_POINT_1D_CASES = ((2, 6, :typed), (3, 300, :raw), (4, 6, :raw))
const DA_JOINT_CASES = ((2, 5), (3, 600), (4, 5))
const DA_ANGLE_CASES = ((2, "device", DA_DEV), (3, "threaded", CB.ThreadedBackend()), (4, "device", DA_DEV))
const DA_SINGLE_PASS_CASES = ((2, :typed, :sp1d), (3, :raw, :sp2d), (4, :typed, :sp2d))
const DA_SLICE_CASES = ((2, true, :typed), (3, true, :raw), (4, false, :raw))
const DA_WIDE_CASES = ((:batch1d, true, false), (:sp1d, false, true), (:joint, true, true), (:sp2d, false, false))

# Each route agrees with serial over a covering of the axes it dispatches on.
Test.@testset "the device agrees across every axis a route dispatches on" begin
    Random.seed!(9100)

    Test.@testset "point 1D: width x bin spelling x bin count" begin
        for (D, nb, spelling) in DA_POINT_1D_CASES
            x, u = rand(D, DA_N), randn(D, DA_N)
            bins = da_bins(spelling, 0.0, 1.0, nb)
            ref = SFC.calculate_structure_function(DA_OP, x, u, collect(bins), Float64, DA_RAW;
                backend = DA_SER)
            got = SFC.calculate_structure_function(DA_OP, x, u, bins, Float64, DA_RAW;
                backend = DA_DEV)
            Test.@test sum(ref.counts) > 0
            Test.@test (D, nb, spelling, da_agrees(got.sums, got.counts, ref.sums, ref.counts)) ==
                       (D, nb, spelling, true)
        end
    end

    Test.@testset "joint 2D: width x bin count across the shared-memory cap" begin
        for (D, nv) in DA_JOINT_CASES
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

    # Binning by angle differs from binning by value, and every backend matches serial's angle histogram.
    Test.@testset "the second axis x width: the angle is a property of the metric" begin
        dbins = collect(range(0.0, 1.0; length = 7))
        abins = collect(range(prevfloat(0.0), Float64(π); length = 5))
        for (D, name, be) in DA_ANGLE_CASES
            x, u = rand(D, DA_N), randn(D, DA_N)
            ax = SFC.SeparationAngleAxis(SA.SVector(ntuple(d -> d == 1 ? 1.0 : 0.0, D)))
            ref = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                backend = DA_SER, second_axis = ax)
            Test.@test sum(ref.counts) > 0
            val = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                backend = DA_SER)
            Test.@test collect(ref.counts) != collect(val.counts)
            got = SFC.calculate_structure_function(DA_OP, x, u, dbins, abins;
                backend = be, second_axis = ax)
            Test.@test (D, name, da_agrees(got.sums, got.counts, ref.sums, ref.counts)) ==
                       (D, name, true)
        end
    end

    Test.@testset "single-pass 1D and 2D: width x bin spelling" begin
        for (D, spelling, family) in DA_SINGLE_PASS_CASES
            x, u = rand(D, DA_N), randn(D, DA_N)
            vbins = collect(range(-3.0, 3.0; length = 6))
            bins = da_bins(spelling, 0.0, 1.0, 8)
            nb = length(bins) - 1
            if family === :sp1d
                rs = zeros(SFC.SINGLE_PASS_N, nb); rc = zeros(Int, SFC.SINGLE_PASS_N, nb)
                SFC.calculate_structure_functions_single_pass!(rs, rc, x, u, collect(bins);
                    backend = DA_SER)
                gs = zeros(SFC.SINGLE_PASS_N, nb); gc = zeros(Int, SFC.SINGLE_PASS_N, nb)
                SFC.calculate_structure_functions_single_pass!(gs, gc, x, u, bins;
                    backend = DA_DEV)
                Test.@test (D, spelling, :sp1d, da_agrees(gs, gc, rs, rc)) ==
                           (D, spelling, :sp1d, true)
            elseif family === :sp2d
                r2s = zeros(SFC.SINGLE_PASS_N, nb, 5); r2c = zeros(Int, SFC.SINGLE_PASS_N, nb, 5)
                SFC.calculate_structure_functions_single_pass_2d!(r2s, r2c, x, u, collect(bins),
                    vbins; backend = DA_SER)
                g2s = zeros(SFC.SINGLE_PASS_N, nb, 5); g2c = zeros(Int, SFC.SINGLE_PASS_N, nb, 5)
                SFC.calculate_structure_functions_single_pass_2d!(g2s, g2c, x, u, bins, vbins;
                    backend = DA_DEV)
                Test.@test (D, spelling, :sp2d, da_agrees(g2s, g2c, r2s, r2c)) ==
                           (D, spelling, :sp2d, true)
            else
                error("unknown single-pass family $family")
            end
        end
    end

    Test.@testset "slice batch: width x bin spelling x shared or varying positions" begin
        T = 3
        for (D, shared, spelling) in DA_SLICE_CASES
            u = randn(D, DA_N, T)
            x = shared ? rand(D, DA_N) : rand(D, DA_N, T)
            bins = da_bins(spelling, 0.0, 1.0, 8)
            nb = length(bins) - 1
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

    # Four Float64 tiles of 13 coordinates exceed every device's static shared memory.
    Test.@testset "slice batch at a width no staged tile holds" begin
        D, T = 13, 3
        dbins = collect(range(0.0, 2.0; length = 7))
        vbins = collect(range(0.0, 8.0; length = 6))
        for (name, shared, weighted) in DA_WIDE_CASES
            u = randn(D, DA_N, T)
            x = shared ? rand(D, DA_N) : rand(D, DA_N, T)
            kw = weighted ? (; weights = rand(DA_N)) : (;)
            CT = weighted ? Float64 : Int
            shape, run!, slice! = da_batch_family(name, x, u, dbins, vbins, T, kw)
            rs, rc = zeros(shape), zeros(CT, shape)
            for t in 1:T
                s, c = zeros(shape[1:(end - 1)]), zeros(CT, shape[1:(end - 1)])
                slice!(s, c, shared ? x : x[:, :, t], u[:, :, t])
                selectdim(rs, length(shape), t) .= s
                selectdim(rc, length(shape), t) .= c
            end
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
