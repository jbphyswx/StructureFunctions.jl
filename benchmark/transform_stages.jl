# Where a gridded call's time goes, per schedule, and how the two exact algorithms compare.
#
# Gated, not part of the suite: it spends minutes of wall clock and needs reserved cores to mean
# anything. Ratios and shares measured inside one job are portable; absolute seconds are not, so
# nothing here pins a time.
#
# The load-bearing assertion is that **the transform and the direct lag sweep agree on every
# schedule profiled**. Both are exact, so a stage table taken from a route that has started
# returning a different answer is worthless, and this is the one check here that can fail for a
# reason: it needs no tuned constant, because the counts must be equal and the sums equal to
# round-off whatever the data is.
#
# Run: julia -t 8 --project=test benchmark/transform_stages.jl

using Printf: Printf
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW

"""Minimum over `reps` runs: on a shared node a single sample carries noise the size of the effect."""
function fastest(f, reps::Int = 5)
    f()
    return minimum(begin
        t0 = time()
        f()
        time() - t0
    end for _ in 1:reps)
end

const OP = SFT.L2SFType()
const TAG = SB.FastFourierTransformSpectralBackend()

"""One schedule: the forward stage, the whole transform, and the direct sweep over the same data."""
function profile_schedule(name, s, data, bins; failures)
    nb = SFC.n_histogram_bins(SF.BinEdges(bins))
    sums, counts = zeros(nb), zeros(Int, nb)
    vD, vV, vK = Val(2), Val(1), Val(0)

    engine() = SFC.transform_engine(OP, data, s, SF.BinEdges(bins), vD, vV, vK,
                                    SFC.AllValid(), SFC.NoWeights(), TAG)
    transform() = (fill!(sums, 0); fill!(counts, 0);
        SFC.gridded_sweep!(sums, counts, OP, data, s, bins, vD, vV, vK, TAG))
    sweep() = (fill!(sums, 0); fill!(counts, 0);
        SFC.gridded_lag_sweep!(sums, counts, OP, data, s, bins, vD, vV, vK))

    t_fwd = fastest(engine)
    t_tr = fastest(transform)
    t_sw = fastest(sweep)

    # the two routes are exact and must agree; a stage table of a wrong answer is worthless
    transform(); a_s, a_c = copy(sums), copy(counts)
    sweep()
    agree = a_c == counts && isapprox(a_s, sums; rtol = 1e-9)
    agree || push!(failures, "$name: the transform and the sweep disagree")

    share = t_fwd / t_tr

    Printf.@printf("%-28s forward %7.4f (%4.1f%%)  transform %7.4f  sweep %7.4f (%5.2f× the transform)  %s\n",
        name, t_fwd, 100share, t_tr, t_sw, t_sw / t_tr, agree ? "agree" : "DISAGREE")
    return nothing
end

function main()
    Random.seed!(20260918)
    println("threads=", Threads.nthreads(), "  FFTW threads=", FFTW.get_num_threads())
    failures = String[]

    for n in (128, 256)
        s = SFC.UniformLagSchedule((n, n), (2π / n, 2π / n), (true, true))
        u = reshape(Random.randn(2, n, n), 2, :)
        bins = collect(range(0.0, 2.0; length = 33))
        profile_schedule("uniform $(n)² periodic", s, u, bins; failures)
    end

    # a bounded grid pads, so the forward stage writes padding here
    let n = 128
        s = SFC.UniformLagSchedule((n, n), (1 / n, 1 / n), (false, false))
        u = reshape(Random.randn(2, n, n), 2, :)
        bins = collect(range(0.0, 0.5; length = 25))
        profile_schedule("uniform $(n)² bounded", s, u, bins; failures)
    end

    # many slabs: the regime where the per-pair work, not the transforms, sets the cost
    let (nlon, nlat) = (180, 90)
        lats = collect(range(-π / 2 + π / (2nlat), π / 2 - π / (2nlat); length = nlat))
        s = SFC.ZonalLagSchedule(lats, nlon, 2π / nlon, 1.0, true)
        u = reshape(Random.randn(2, nlon, nlat), 2, :)
        bins = collect(range(0.0, π; length = 25))
        profile_schedule("zonal $(nlon)×$(nlat)", s, u, bins; failures)
    end

    println()
    if isempty(failures)
        println("TRANSFORM STAGES OK")
        return 0
    end
    println("TRANSFORM STAGES FAILED: ", join(failures, "; "))
    return 1
end

exit(main())
