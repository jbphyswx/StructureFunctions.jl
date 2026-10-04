# Launch-plan sweeps. A section times every candidate launch of one kernel family over the regimes the family
# serves, checks that each candidate fills the histogram its first candidate fills, writes one CSV row per
# (regime, candidate), and prints for each class the candidate with the least worst-case regret over the class's
# regimes, which is the choice the tree's plan tables hold.
#   julia --project=gpu gpu/benchmark_launch_plans.jl <section> rows.csv [FT W NMOM]
# with <section> one of native1d, native1d_small, fixed1d, fixed1d_large, carveout1d, native2d, native2d_small, fixed2d,
# lag, harmonic, sp2d_portable, batch1d. The optional `FT W NMOM` (each `*` for every value) restrict native1d, fixed1d
# and the native 2-D sections to those classes.
using CUDA: CUDA
using Printf: Printf
using Random: Random
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const CE = Base.get_extension(SF, :StructureFunctionsCUDAExt)
const BE = CUDA.CUDABackend()
const NVDEV = CUDA.NVML.Device(CUDA.uuid(CUDA.device()))

"""The classes `(FT, W, NMOM)` the command line selects, each `*` for every value."""
const ONLY = (get(ARGS, 3, "*"), get(ARGS, 4, "*"), get(ARGS, 5, "*"))
selected(FT, W, NMOM) = all(((o, v),) -> o == "*" || o == string(v), zip(ONLY, (FT, W, NMOM)))

"""The SM clock now, in MHz."""
sm_mhz() = CUDA.NVML.clock_info(NVDEV).sm

"""Run `f` to completion, call after call, for `seconds`, so the SM clock has left the state an idle device drops to
before anything is timed."""
function hold_clock(f, seconds = 0.5)
    t0 = time()
    while time() - t0 < seconds
        f()
        CUDA.synchronize()
    end
    return nothing
end

"""Best of `reps` device-timed runs of `f`, in ms, after one untimed run, and the SM clock after them in MHz."""
function best_ms(f; reps = 5)
    f()
    CUDA.synchronize()
    t = 1e3 * minimum(_ -> CUDA.@elapsed(f()), 1:reps)
    return t, sm_mhz()
end

"""Each `(candidate, run)`'s best time in ms over a forward and a reverse pass after holding the clock up, so no
candidate's time depends on where in the order it ran, with the SM clock of its best pass: `c => (ms, MHz)`."""
function timed(cands)
    t = Dict(c => (Inf, 0) for (c, _) in cands)
    isempty(cands) && return t
    hold_clock(last(first(cands)))
    for pass in (cands, reverse(cands)), (c, run) in pass
        ms, mhz = best_ms(run)
        ms < first(t[c]) && (t[c] = (ms, mhz))
    end
    return t
end

"""The lowest and highest SM clock, in MHz, over the rows `(class, regime, candidate, ms, MHz)` of each class."""
function clock_span(rows)
    return Dict(class => extrema(r[5] for r in rows if r[1] == class) for class in unique(first.(rows)))
end

"""Rows `(class, regime, candidate, ms)` → per class, the candidate minimising its largest ratio to the best
candidate of each regime, with that ratio; candidates missing from a regime count as infinitely slow there."""
function least_regret(rows)
    out = Dict{Any, Tuple{Any, Float64}}()
    for class in unique(first.(rows))
        cr = filter(r -> r[1] == class, rows)
        best = Dict(reg => minimum(r[4] for r in cr if r[2] == reg) for reg in unique(getindex.(cr, 2)))
        cands = unique(getindex.(cr, 3))
        regret(c) = maximum(reg -> begin
                                t = [r[4] for r in cr if r[2] == reg && r[3] == c]
                                isempty(t) ? Inf : only(t) / best[reg]
                            end, keys(best))
        c = argmin(regret, cands)
        out[class] = (c, regret(c))
    end
    return out
end

"""Per class of the rows `(class, regime, fraction, candidate, ms)`, the split of its regimes by in-range fraction into
at most `K` bands, each served by one candidate, with the least worst-case regret: `class => (regret, [(upper, c)])`,
`upper` the band's highest fraction."""
function banded_regret(rows, K)
    out = Dict{Any, Any}()
    for class in unique(first.(rows))
        cr = filter(r -> r[1] == class, rows)
        regs = sort(unique(r -> r[2], cr); by = r -> r[3])
        best = Dict(r[2] => minimum(q[5] for q in cr if q[2] == r[2]) for r in regs)
        cands = unique(getindex.(cr, 4))
        time_of = Dict((r[2], r[4]) => r[5] for r in cr)
        regret(lo, hi, c) = maximum(k -> get(time_of, (regs[k][2], c), Inf) / best[regs[k][2]], lo:hi)
        n = length(regs)
        choice = (Inf, [])
        for cuts in Iterators.flatten(combinations_upto(n - 1, K - 1))
            bounds = [0; collect(cuts); n]
            parts = [(bounds[i] + 1, bounds[i + 1]) for i in 1:(length(bounds) - 1)]
            cs = [argmin(c -> regret(lo, hi, c), cands) for (lo, hi) in parts]
            w = maximum(regret(lo, hi, c) for ((lo, hi), c) in zip(parts, cs))
            w < first(choice) && (choice = (w, [(regs[hi][3], c) for ((_, hi), c) in zip(parts, cs)]))
        end
        out[class] = choice
    end
    return out
end

"""Every increasing tuple of at most `k` cut positions in `1:m`."""
combinations_upto(m, k) = (Iterators.filter(t -> all(i -> t[i] < t[i + 1], 1:(j - 1)), Iterators.product(ntuple(_ -> 1:m, j)...))
                           for j in 0:min(k, m))

# The native 1-D kernel: `(TILE, R, Q)` — tile, histogram replicas at stride `S = R`, and the queued kernel's ballot
# threshold, 0 for the direct kernel — over bin capacity, the pair fraction in range the launch estimates, moments,
# precision, width, point calls of `points` points and, with `batch`, batch launches of the pairs of a point call of
# 20 000. The summary gives, per class, the plan with the least worst-case regret and the best split into two and three
# bands of in-range fraction.
function native1d(io; points = (20_000,), batch = true)
    caps = SFC.gpu_device_caps(BE)
    Nb, Bb = 5_000, 16
    rows = []
    println(io, "FT,W,NMOM,NB,H,N,rmax,fraction,mode,TILE,R,Q,ms,sm_mhz")
    for FT in (Float32, Float64), W in (2, 3), (NMOM, M) in ((1, SFT.L2SFType()), (6, SFT.SinglePassInvariants())),
        NB in (16, 32, 64, 128)
        selected(FT, W, NMOM) || continue
        H = SFC._val_int(CE._cuda_1d_bin_capacity(NB))
        geom = SF.HelperFunctions.FlatGeometry{W}()
        rng = Random.Xoshiro(NB + 7W)
        modes = Any[("point", Np, CUDA.CuArray(rand(rng, FT, W, Np)), CUDA.CuArray(randn(rng, FT, W, Np, 1)), 1, false)
                    for Np in points]
        batch && push!(modes, ("batch", Nb, CUDA.CuArray(rand(rng, FT, W, Nb)), CUDA.CuArray(randn(rng, FT, W, Nb, Bb)),
                               Bb, true))
        for rmax in (0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5), (mode, N, x, u, B, fixed) in modes
            dig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(rmax), NB + 1), Val(NMOM == 1 ? :sf1d : :single_pass))
            f = SFC.gpu_in_range_fraction(BE, x, dig, NB, geom, nothing, GE.SF_GPU_TILE)
            out, cnt = CUDA.zeros(FT, NMOM, NB, B), CUDA.zeros(UInt32, NMOM, NB, B)
            ref = nothing
            cands = []
            for TILE in (128, 256, 384, 512), R in (1, 2, 4, 8, 16, 32), Q in (NMOM == 1 ? (0,) : (0, 24))
                SFC.gpu_static_smem_fits(caps, CE._cuda_1d_smem_bytes(FT, FT, FT, UInt32, W, W, NMOM, TILE, R, H, Q)) ||
                    continue
                plan = CE.CUDA1DPlan{W, W, NMOM, TILE, R, R, H, UInt32, Q}()
                run = () -> SFC.gpu_native_launch_1d!(plan, out, cnt, x, u, SFC.NoWeights(), M, dig, N, NB, B,
                                                      fixed, geom, nothing)
                fill!(out, 0); fill!(cnt, 0)
                run()
                got = Array(cnt)
                ref === nothing && (ref = got)
                got == ref || error("native1d: TILE=$TILE R=$R Q=$Q counts differ from the first candidate at " *
                                    "$FT W=$W NMOM=$NMOM NB=$NB rmax=$rmax $mode")
                push!(cands, ((TILE, R, Q), run))
            end
            for ((TILE, R, Q), (t, mhz)) in sort(collect(timed(cands)); by = first)
                println(io, join((FT, W, NMOM, NB, H, N, rmax, Printf.@sprintf("%.4f", f), mode, TILE, R, Q,
                                  Printf.@sprintf("%.4f", t), mhz), ","))
                push!(rows, ((NMOM, H, FT, W), (NB, rmax, mode, N), f, (TILE, R, Q), t, mhz))
            end
            flush(io)
        end
    end
    clocks = clock_span([(r[1], r[2], r[4], r[5], r[6]) for r in rows])
    banded = [banded_regret([r[1:5] for r in rows], K) for K in 1:3]
    println("\nnative 1-D per (NMOM, H, FT, W): least-regret plan, and the best split into 2 and 3 fraction bands")
    for class in sort(unique(first.(rows)); by = string)
        Printf.@printf("  %-28s SM %d-%d MHz\n", class, clocks[class]...)
        for K in 1:3
            w, bands = banded[K][class]
            Printf.@printf("    %d band(s) worst regret %.3f: %s\n", K, w,
                           join(("≤$(Printf.@sprintf("%.3f", up)) $(c)" for (up, c) in bands), ", "))
        end
    end
end

# The native 1-D kernels on a batch over shared positions: per-slice plans `(TILE, R, Q)` against strip plans
# `(:strip, TILE, SW, R)`, over precision, width, moments, bins, point counts `points`, slice counts `slices` and the
# pair fraction in range the launch estimates. The summary gives, per class, point count and slice count, the plan with
# the least worst-case regret and the best split into two and three fraction bands.
function fixed1d(io; points = (5_000,), slices = (4, 8, 16, 32))
    caps = SFC.gpu_device_caps(BE)
    rows = []
    println(io, "FT,W,NMOM,NB,H,N,B,rmax,fraction,plan,ms,sm_mhz")
    for FT in (Float32, Float64), W in (2, 3), (NMOM, M) in ((1, SFT.L2SFType()), (6, SFT.SinglePassInvariants())),
        NB in (16, 32, 64, 128), N in points, B in slices
        selected(FT, W, NMOM) || continue
        H = SFC._val_int(CE._cuda_1d_bin_capacity(NB))
        geom = SF.HelperFunctions.FlatGeometry{W}()
        rng = Random.Xoshiro(NB + 7W + B + N)
        x, u = CUDA.CuArray(rand(rng, FT, W, N)), CUDA.CuArray(randn(rng, FT, W, N, B))
        plans = Any[]
        for TILE in (128, 256, 384, 512), R in (1, 2, 4, 8, 16, 32), Q in (NMOM == 1 ? (0,) : (0, 24))
            SFC.gpu_static_smem_fits(caps, CE._cuda_1d_smem_bytes(FT, FT, FT, UInt32, W, W, NMOM, TILE, R, H, Q)) ||
                continue
            push!(plans, ((TILE, R, Q), CE.CUDA1DPlan{W, W, NMOM, TILE, R, R, H, UInt32, Q}()))
        end
        for TILE in (256, 384), SW in (2, 4, 8, 16), R in (1, 2, 4)
            SW <= B || continue
            SFC.gpu_static_smem_fits(caps, CE._cuda_1d_strip_smem_bytes(FT, FT, FT, UInt32, W, W, NMOM, TILE, SW, R, H)) ||
                continue
            push!(plans, ((:strip, TILE, SW, R), CE.CUDA1DStripPlan{W, W, NMOM, TILE, SW, R, H, UInt32}()))
        end
        for rmax in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.5)
            dig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(rmax), NB + 1), Val(NMOM == 1 ? :sf1d : :single_pass))
            f = SFC.gpu_in_range_fraction(BE, x, dig, NB, geom, nothing, GE.SF_GPU_TILE)
            out, cnt = CUDA.zeros(FT, NMOM, NB, B), CUDA.zeros(UInt32, NMOM, NB, B)
            ref = nothing
            cands = []
            for (spec, plan) in plans
                run = () -> SFC.gpu_native_launch_1d!(plan, out, cnt, x, u, SFC.NoWeights(), M, dig, N, NB, B, true,
                                                      geom, nothing)
                fill!(out, 0); fill!(cnt, 0)
                run()
                got = Array(cnt)
                ref === nothing && (ref = got)
                got == ref || error("fixed1d: $spec counts differ from the first plan at $FT W=$W NMOM=$NMOM NB=$NB " *
                                    "N=$N B=$B rmax=$rmax")
                push!(cands, (spec, run))
            end
            for (spec, (t, mhz)) in sort(collect(timed(cands)); by = string ∘ first)
                println(io, join((FT, W, NMOM, NB, H, N, B, rmax, Printf.@sprintf("%.4f", f), "\"$spec\"",
                                  Printf.@sprintf("%.4f", t), mhz), ","))
                push!(rows, ((NMOM, H, FT, W, N, B), (NB, rmax), f, spec, t, mhz))
            end
            flush(io)
        end
    end
    clocks = clock_span([(r[1], r[2], r[4], r[5], r[6]) for r in rows])
    banded = [banded_regret([r[1:5] for r in rows], K) for K in 1:3]
    println("\nfixed-position batch per (NMOM, H, FT, W, N, B): least-regret plan, and the best split into 2 and 3 fraction bands")
    for class in sort(unique(first.(rows)); by = string)
        Printf.@printf("  %-28s SM %d-%d MHz\n", class, clocks[class]...)
        for K in 1:3
            w, bs = banded[K][class]
            Printf.@printf("    %d band(s) worst regret %.3f: %s\n", K, w,
                           join(("≤$(Printf.@sprintf("%.3f", up)) $(c)" for (up, c) in bs), ", "))
        end
    end
end

"""The compiled `_cuda_sf_1d_kernel!` that the direct plan `plan`'s launch runs for these arguments."""
function cuda_1d_kernel(::CE.CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST, 0}, out, cnt, x, u, M, dig, N, NB, B, fixed,
                        geom) where {W, F, NMOM, TILE, R, S, H, CST}
    xv = fixed ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(nothing, N, TILE)
    return CUDA.@cuda launch=false CE._cuda_sf_1d_kernel!(
        out, cnt, xv, uv, SFC.NoWeights(), M, dig, N, NB, sched, SFC.n_pair_blocks(sched), Val(W), Val(F),
        Val(NMOM), Val(fixed), Val(TILE), Val(R), Val(S), Val(H), Val(CST), geom)
end

# The native 1-D candidates at the device's default shared-memory carveout and at the largest shared carveout,
# over a subset of the native1d regimes: one row per (regime, candidate) with both times.
function carveout1d(io)
    caps = SFC.gpu_device_caps(BE)
    N, Bb, W = 20_000, 16, 2
    geom = SF.HelperFunctions.FlatGeometry{W}()
    rows = []
    println(io, "FT,NMOM,NB,frac,mode,TILE,R,ms_default,sm_mhz_default,ms_maxshared,sm_mhz_maxshared")
    for FT in (Float32, Float64), (NMOM, M) in ((1, SFT.L2SFType()), (6, SFT.SinglePassInvariants())), NB in (32, 64)
        H = SFC._val_int(CE._cuda_1d_bin_capacity(NB))
        rng = Random.Xoshiro(NB + 7W)
        x = CUDA.CuArray(rand(rng, FT, W, N))
        u1 = CUDA.CuArray(randn(rng, FT, W, N, 1))
        ub = CUDA.CuArray(randn(rng, FT, W, N, Bb))
        for frac in (0.05, 0.3, 1.5), (mode, u, B, fixed) in (("point", u1, 1, false), ("batch", ub, Bb, true))
            dig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(frac), NB + 1), Val(NMOM == 1 ? :sf1d : :single_pass))
            out, cnt = CUDA.zeros(FT, NMOM, NB, B), CUDA.zeros(UInt32, NMOM, NB, B)
            cands, kerns = [], []
            for TILE in (128, 256, 384, 512, 1024), R in (1, 2, 4, 8, 16, 32)
                SFC.gpu_static_smem_fits(caps, CE._cuda_1d_smem_bytes(FT, FT, FT, UInt32, W, W, NMOM, TILE, R, H, 0)) ||
                    continue
                plan = CE.CUDA1DPlan{W, W, NMOM, TILE, R, R, H, UInt32, 0}()
                run = () -> SFC.gpu_native_launch_1d!(plan, out, cnt, x, u, SFC.NoWeights(), M, dig, N, NB, B,
                                                      fixed, geom, nothing)
                run()
                push!(cands, ((TILE, R), run))
                push!(kerns, cuda_1d_kernel(plan, out, cnt, x, u, M, dig, N, NB, B, fixed, geom))
            end
            carve!(p) = foreach(k -> (CUDA.attributes(k.fun)[CUDA.FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = p),
                                kerns)
            carve!(-1)
            td = timed(cands)
            carve!(100)
            tm = timed(cands)
            carve!(-1)
            for ((TILE, R), _) in cands
                (ad, cd), (am, cm) = td[(TILE, R)], tm[(TILE, R)]
                println(io, join((FT, NMOM, NB, frac, mode, TILE, R, Printf.@sprintf("%.4f", ad), cd,
                                  Printf.@sprintf("%.4f", am), cm), ","))
                push!(rows, ((NMOM, H, FT), (NB, frac, mode), (TILE, R, 0), ad, cd))
                push!(rows, ((NMOM, H, FT), (NB, frac, mode), (TILE, R, 100), am, cm))
            end
            flush(io)
        end
    end
    clocks = clock_span(rows)
    println("\nnative 1-D least-regret (TILE, R, carveout %) per (NMOM, H, FT), with the class's SM clock span:")
    for (class, (c, r)) in sort(collect(least_regret(rows)); by = string ∘ first)
        Printf.@printf("  %-20s TILE=%-5d R=%-3d carveout=%-3d worst regret %.3f  SM %d-%d MHz\n", class, c..., r,
                       clocks[class]...)
    end
end

# The native 2-D kernels: the shared-histogram kernel at its tile and planes per launch `NP` (0 marks the global-atomic
# kernel), over histogram shape, the pair fraction in range the launch estimates, moments, precision, width, the value
# edges' digitizer (`linear`: `LinearBinEdges`, computed; `lookup`: a vector of edges, looked up), point calls of
# `points` points and batches over shared positions of each `(N, B)` of `batches`, over `shapes` and cutoffs `rmaxes`.
# Each candidate is its own compiled kernel and may round a pair's value differently in the last bit, so a value on a
# value edge can land in the neighbouring bin; the counts check allows 1e-4 of the counts to move. The summary gives,
# per (class, shape), the plan with the least worst-case regret and the best split into two and three fraction bands.
function native2d(io; points = (20_000,), batches = ((5_000, 16),),
                  shapes = ((16, 8), (32, 16), (50, 50), (64, 64), (96, 96), (128, 192)),
                  rmaxes = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0, 1.5))
    caps = SFC.gpu_device_caps(BE)
    rows = []
    println(io, "FT,W,NMOM,values,shape,cells,N,B,rmax,fraction,mode,TILE,NP,ms,sm_mhz")
    for FT in (Float32, Float64), W in (2, 3), (NMOM, M) in ((1, SFT.L2SFType()), (6, SFT.SinglePassInvariants())),
        values in (:linear, :lookup), (nd, nv) in shapes
        selected(FT, W, NMOM) || continue
        geom = SF.HelperFunctions.FlatGeometry{W}()
        rng = Random.Xoshiro(nd + nv + 7W)
        modes = Any[("point", Np, CUDA.CuArray(rand(rng, FT, W, Np)), CUDA.CuArray(randn(rng, FT, W, Np, 1)), 1, false)
                    for Np in points]
        for (Nb, Bb) in batches
            push!(modes, ("batch", Nb, CUDA.CuArray(rand(rng, FT, W, Nb)), CUDA.CuArray(randn(rng, FT, W, Nb, Bb)), Bb, true))
        end
        vbins = values === :linear ? SF.LinearBinEdges(FT(-1), FT(3), nv + 1) : collect(range(FT(-1), FT(3); length = nv + 1))
        vplan = GE._value_digitizer(nothing, BE, vbins)
        hcells = nd * CE._cuda_val_stride(nv)
        for rmax in rmaxes, (mode, N, x, u, B, fixed) in modes
            ddig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(rmax), nd + 1),
                                     Val(NMOM == 1 ? :joint2d : :single_pass_2d))
            f = SFC.gpu_in_range_fraction(BE, x, ddig, nd, geom, nothing, GE.SF_GPU_TILE)
            out, cnt = CUDA.zeros(FT, NMOM, nd, nv, B), CUDA.zeros(UInt32, NMOM, nd, nv, B)
            ref = nothing
            cands = []
            for TILE in (128, 256, 512, 1024), NP in (0, 1, 2, 3, 6)
                NP <= NMOM || continue
                staging = 4 * SFC.gpu_localmem_bytes(FT, W * TILE)
                SFC.gpu_static_smem_fits(caps, staging) || continue
                cells = NP * hcells
                dynb = CE._cuda_count_plane_offset(FT, cells) + cells * sizeof(UInt32)
                NP == 0 || staging + dynb <= caps.smem_optin || continue
                plan = NP == 0 ? CE.CUDA2DGlobalPlan{W, W, NMOM, TILE}() :
                       CE.CUDA2DPlan{W, W, NMOM, TILE, UInt32, NP}(hcells, dynb)
                run = () -> SFC.gpu_native_launch_2d!(plan, out, cnt, x, u, SFC.NoWeights(), M, ddig, vplan, N, nd,
                                                      nv, B, fixed, geom, SFC.InvariantValueAxis(), nothing, nothing)
                fill!(out, 0); fill!(cnt, 0)
                run()
                got = Array(cnt)
                ref === nothing && (ref = got)
                moved = sum(abs, Int64.(got) .- Int64.(ref)) ÷ 2
                moved <= max(10, sum(Int64, ref) ÷ 10^4) ||
                    error("native2d: TILE=$TILE NP=$NP moves $moved pairs from the first candidate at " *
                          "$FT W=$W NMOM=$NMOM $values $(nd)x$(nv) rmax=$rmax $mode")
                push!(cands, ((TILE, NP), run))
            end
            for ((TILE, NP), (t, mhz)) in sort(collect(timed(cands)); by = first)
                println(io, join((FT, W, NMOM, values, "$(nd)x$(nv)", hcells, N, B, rmax, Printf.@sprintf("%.4f", f), mode,
                                  TILE, NP, Printf.@sprintf("%.4f", t), mhz), ","))
                push!(rows, ((NMOM, FT, W, values, "$(nd)x$(nv)"), (rmax, mode, N, B), f, (TILE, NP), t, mhz))
            end
            flush(io)
        end
    end
    clocks = clock_span([(r[1], r[2], r[4], r[5], r[6]) for r in rows])
    banded = [banded_regret([r[1:5] for r in rows], K) for K in 1:3]
    println("\nnative 2-D per (NMOM, FT, W, shape): least-regret (TILE, NP), and the best split into 2 and 3 fraction bands")
    for class in sort(unique(first.(rows)); by = string)
        Printf.@printf("  %-34s SM %d-%d MHz\n", class, clocks[class]...)
        for K in 1:3
            w, bands = banded[K][class]
            Printf.@printf("    %d band(s) worst regret %.3f: %s\n", K, w,
                           join(("≤$(Printf.@sprintf("%.3f", up)) $(c)" for (up, c) in bands), ", "))
        end
    end
end

# The device lag sweep: workgroup lanes `WG`, lanes `L` per (slab pair, lag) and chunks per item, each lane sweeping
# 8 to 1024 of a slab's cells, over grid, lag reach (in cells), operator and batch size. Each row records the regime's
# items, slices and slab cells, the arguments of the plan rule, and whether the candidate is the rule's plan; the
# least-regret rule constants follow from the rows.
function lag(io)
    DEV = CB.GPUBackend(BE)
    rows = []
    println(io, "sched,cells,n_items,nt,rcells,op,WG,L,n_chunks,rule,ms,sm_mhz")
    zonal(nlon, nlat) = SFC.ZonalLagSchedule(collect(range(-80.0, 80.0; length = nlat)) .* (π / 180), nlon, 2π / nlon, 1.0,
                                             true)
    for (sname, s, h) in (("uniform128", SFC.UniformLagSchedule((128, 128), (1 / 128, 1 / 128), (true, true)), 1 / 128),
                          ("uniform256", SFC.UniformLagSchedule((256, 256), (1 / 256, 1 / 256), (true, true)), 1 / 256),
                          ("uniform512", SFC.UniformLagSchedule((512, 512), (1 / 512, 1 / 512), (true, true)), 1 / 512),
                          ("zonal360x180", zonal(360, 180), 2π / 360), ("zonal720x360", zonal(720, 360), 2π / 720)),
        (oname, op) in (("L2", SFT.L2SFType()), ("S3", SFT.S3SFType())), rcells in (4, 12, 32), nt in (1, 4)
        n = SFC.n_cells(s)
        data = randn(Random.Xoshiro(n + rcells + nt), 2, n, nt)
        edges = collect(range(0.0, rcells * h; length = 33))
        plan = SFC.squared_digitize_plan(edges)
        nb = SFC.n_histogram_bins(plan)
        su = SFC.uniform_axes(s)
        laid, lv, lw = SFC._batch_layout(s, data, SFC.AllValid(), SFC.NoWeights())
        sums, counts = CUDA.zeros(Float64, nb, nt), CUDA.zeros(Int64, nb, nt)
        args = (DEV, op, s, su, laid, lv, lw, plan, nb, float(last(edges)), SFC.lag_transport(s), nothing, Val(2), Val(1),
                Val(0))
        seen = Ref((0, 0, 0))
        GE._device_lag_sweep!((caps, n_items, t, cells) -> (seen[] = (n_items, t, cells); GE._lag_plan(caps, n_items, t, cells)),
                              sums, counts, args...)
        n_items, _, cells = seen[]
        rule_plan = GE._lag_plan(SFC.gpu_device_caps(BE), n_items, nt, cells)
        ref = Array(counts)
        cands = []
        for WG in (128, 256, 512), L in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512), nc in (1, 2, 4, 8, 16)
            (L <= WG && 8 <= cells / (L * nc) <= 1024) || continue
            lp = GE.LagPlan{WG, L}(nc)
            run = () -> GE._device_lag_sweep!((_...) -> lp, sums, counts, args...)
            fill!(sums, 0); fill!(counts, 0)
            run()
            Array(counts) == ref || error("lag: WG=$WG L=$L n_chunks=$nc counts differ from the rule's at $sname $oname " *
                                          "rcells=$rcells nt=$nt")
            push!(cands, ((WG, L, nc), run))
        end
        rc = (typeof(rule_plan).parameters[1], typeof(rule_plan).parameters[2], rule_plan.n_chunks)
        for ((WG, L, nc), (t, mhz)) in sort(collect(timed(cands)); by = first)
            println(io, join((sname, cells, n_items, nt, rcells, oname, WG, L, nc, (WG, L, nc) == rc, Printf.@sprintf("%.4f", t),
                              mhz), ","))
            push!(rows, ((sname, oname, rcells, nt), (WG, L, nc) == rc, t, (WG, L, nc)))
        end
        flush(io)
    end
    println("\nlag sweep: the rule's plan against the best candidate per regime:")
    for reg in unique(first.(rows))
        rr = filter(r -> r[1] == reg, rows)
        b = argmin(r -> r[3], rr)
        rl = filter(r -> r[2], rr)
        Printf.@printf("  %-34s best WG=%-4d L=%-4d chunks=%-3d %8.3f ms   rule %s %.3f\n", reg, b[4]..., b[3],
                       isempty(rl) ? "-" : string(only(rl)[4]), isempty(rl) ? NaN : only(rl)[3] / b[3])
    end
end

# The device harmonic coefficients: workgroup lanes `WG`, points per lane `PTS`, degree tile `LT` and workgroups per
# multiprocessor, over point count, band limit and spin; `WG` a multiple of a lane's `4·LT` values and the kernel's
# static shared memory within the budget.
function harmonic(io)
    caps = SFC.gpu_device_caps(BE)
    rows = []
    println(io, "N,lmax,s,WG,PTS,LT,groups_per_sm,n_chunks,rule,ms,sm_mhz")
    for N in (20_000, 200_000), lmax in (32, 128), s in (0, 1)
        rng = Random.Xoshiro(N + lmax + s)
        f = CUDA.CuArray(complex.(randn(rng, N), randn(rng, N)))
        θ = CUDA.CuArray(acos.(2 .* rand(rng, N) .- 1))
        φ = CUDA.CuArray(2π .* rand(rng, N))
        n_m = s == 0 ? lmax + 1 : 2lmax + 1
        rp = GE._harm_plan(caps, n_m, N)
        rc = (typeof(rp).parameters..., rp.n_chunks)
        ref = nothing
        cands = []
        for WG in (64, 128, 256), LT in (4, 8, 16), PTS in (1, 2, 4, 8), G in (2, 4, 8, 16, 32)
            (WG % (4LT) == 0 && SFC.gpu_static_smem_fits(caps, 8 * ((4LT + 1) * WG + WG + 4LT))) || continue
            nc = clamp(cld(G * caps.n_sms, n_m), 1, max(1, cld(N, WG * PTS)))
            any(c -> c[1][1:3] == (WG, PTS, LT) && c[1][5] == nc, cands) && continue
            hp = GE.HarmPlan{WG, PTS, LT}(nc)
            run = () -> GE._harm_coefficients((_...) -> hp, BE, f, θ, φ, s, lmax)
            got = Array(run())
            ref === nothing && (ref = got)
            maximum(abs.(got .- ref)) <= 1e-10 * maximum(abs, ref) ||
                error("harmonic: WG=$WG PTS=$PTS LT=$LT n_chunks=$nc differs from the first candidate at N=$N lmax=$lmax s=$s")
            push!(cands, ((WG, PTS, LT, G, nc), run))
        end
        for (c, (t, mhz)) in sort(collect(timed(cands)); by = first)
            isrule = (c[1], c[2], c[3], c[5]) == rc
            println(io, join((N, lmax, s, c..., isrule, Printf.@sprintf("%.4f", t), mhz), ","))
            push!(rows, ((N, lmax, s), c, t, isrule))
        end
        flush(io)
    end
    println("\nharmonic: best candidate per regime and the rule's plan against it:")
    for reg in unique(first.(rows))
        rr = filter(r -> r[1] == reg, rows)
        b = argmin(r -> r[3], rr)
        cur = filter(r -> r[4], rr)
        Printf.@printf("  %-22s best WG=%-4d PTS=%-2d LT=%-3d G=%-3d chunks=%-4d %9.3f ms   rule %.3f\n", reg, b[2]...,
                       b[3], isempty(cur) ? NaN : first(cur)[3] / b[3])
    end
end

# The portable SP2D kernels: the accumulation strategy's on-chip mode for the histogram (`:shared`, or `:typeplane`
# over its passes) against the global-atomic kernel at several workgroup sizes, over histogram shape, point count,
# precision and pair fraction in range. Each row records the mode, the typeplane passes and the joint histogram's bytes.
function sp2d_portable(io)
    caps = SFC.gpu_device_caps(BE)
    W = 2
    geom = SF.HelperFunctions.FlatGeometry{W}()
    rows = []
    println(io, "FT,shape,N,mode,passes,hist_bytes,frac,candidate,ms,sm_mhz")
    for FT in (Float32, Float64), (nd, nv) in ((16, 16), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96)),
        N in (2000, 5000, 20_000)
        config = GE._sp2d_accumulation_strategy(caps, nd, nv, W, W, FT, FT, UInt32)
        config === nothing && continue
        mode = config.accum_mode
        bytes = GE._sp2d_joint_cells(nd, nv) * (sizeof(FT) + sizeof(UInt32))
        rng = Random.Xoshiro(nd + nv)
        x, u = CUDA.CuArray(rand(rng, FT, W, N)), CUDA.CuArray(randn(rng, FT, W, N))
        vplan = GE._value_digitizer(nothing, BE, collect(range(FT(-1), FT(3); length = nv + 1)))
        for frac in (0.3, 1.5)
            ddig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(frac), nd + 1), Val(:single_pass_2d))
            s, c = CUDA.zeros(FT, 6, nd, nv), CUDA.zeros(UInt32, 6, nd, nv)
            cands = Any[(string(mode), () -> GE._launch_single_pass_2d_strategy!(BE, s, c, x, u, ddig, vplan, N, nd + 1,
                                                                                 nv + 1, nd, config, geom))]
            for wg in (64, 128, 256)
                push!(cands, ("global$wg", () -> GE._launch_single_pass_2d_kernel!(BE, wg, s, c, x, u, ddig, vplan, N, nd + 1,
                                                                                  nv + 1, geom)))
            end
            ref = nothing
            for (name, run) in cands
                fill!(s, 0); fill!(c, 0)
                run()
                got = Array(c)
                ref === nothing && (ref = got)
                got == ref || error("sp2d_portable: $name counts differ from $mode at $FT $(nd)x$(nv) N=$N frac=$frac")
            end
            for (name, (t, mhz)) in sort(collect(timed(cands)); by = first)
                println(io, join((FT, "$(nd)x$(nv)", N, mode, config.n_type_passes, bytes, frac, name,
                                  Printf.@sprintf("%.4f", t), mhz), ","))
                push!(rows, ((FT, nd, nv, N, frac), (mode, config.n_type_passes, bytes), name, t))
            end
            flush(io)
        end
    end
    println("\nsp2d_portable: per regime, the strategy's mode against the best global-atomic workgroup:")
    for reg in unique(first.(rows))
        rr = filter(r -> r[1] == reg, rows)
        d = only(filter(r -> !startswith(r[3], "global"), rr))
        g = argmin(r -> r[4], filter(r -> startswith(r[3], "global"), rr))
        Printf.@printf("  %-32s %-9s passes %d hist %8d B  mode %8.3f ms  %s %8.3f ms  global/mode %.3f\n", reg,
                       d[2]..., d[4], g[3], g[4], g[4] / d[4])
    end
end

# The 1-D batch over shared positions, unweighted and two-wide: the two-wide field-strip kernel and the portable
# fixed-position strip kernel (each with the staging a call pays) against the native kernel's plan, over precision,
# bins, slices and the pair fraction in range the launch estimates. The summary scores the rules "a strip kernel iff
# the fraction is at most f and the slices at least b" by their worst-case regret.
function batch1d(io)
    N, W = 20_000, 2
    geom = SF.HelperFunctions.FlatGeometry{W}()
    M = SFT.L2SFType()
    rows = []
    println(io, "FT,NB,B,rmax,fraction,candidate,ms,sm_mhz")
    for FT in (Float32, Float64), NB in (16, 64, 128), B in (4, 8, 16, 32, 64)
        rng = Random.Xoshiro(NB + B)
        x = CUDA.CuArray(rand(rng, FT, W, N)); u = CUDA.CuArray(randn(rng, FT, W, N, B))
        plan = SFC.gpu_native_1d_plan(BE, FT, FT, FT, UInt32, SFC.NoWeights(), geom, NB, M)
        for rmax in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 1.5)
            dig = GE._gpu_digitizer(BE, SF.LinearBinEdges(zero(FT), FT(rmax), NB + 1), Val(:sf1d))
            f = SFC.gpu_in_range_fraction(BE, x, dig, NB, geom, nothing, GE.SF_GPU_TILE)
            bufs = [(CUDA.zeros(FT, 1, NB, B), CUDA.zeros(UInt32, 1, NB, B)) for _ in 1:3]
            strip = () -> begin
                xd, ud = GE._stage_batch_device(BE, x, u; fixed_x = true)
                GE._launch_batch_fixed_x_sf!(BE, reshape(bufs[1][1], NB, B), reshape(bufs[1][2], NB, B), xd, ud, M, N, B,
                                             dig, NB, geom)
            end
            fixed = () -> GE._launch_sf_tiled_1d_fixed!(BE, bufs[2]..., x, u, M, dig, N, NB, B, geom)
            native = () -> SFC.gpu_native_launch_1d!(plan, bufs[3]..., x, u, SFC.NoWeights(), M, dig, N, NB, B, true, geom,
                                                     nothing)
            cands = Any[("strip", strip), ("fixed", fixed), ("native", native)]
            for (k, (_, run)) in enumerate(cands)
                fill!.(bufs[k], 0)
                run()
            end
            ref = Array(bufs[3][2])
            all(k -> Array(bufs[k][2]) == ref, 1:2) || error("batch1d: counts differ at $FT NB=$NB B=$B rmax=$rmax")
            for (name, (t, mhz)) in sort(collect(timed(cands)); by = first)
                println(io, join((FT, NB, B, rmax, Printf.@sprintf("%.4f", f), name, Printf.@sprintf("%.4f", t), mhz),
                                 ","))
                push!(rows, ((FT, NB, B, rmax), f, name, t))
            end
            flush(io)
        end
    end
    regimes = unique(r -> r[1], rows)
    time_of(reg, name) = only(r[4] for r in rows if r[1] == reg[1] && r[3] == name)
    best(reg) = minimum(r[4] for r in rows if r[1] == reg[1])
    worst(choose) = maximum(reg -> time_of(reg, choose(reg)) / best(reg), regimes)
    println("\nbatch1d: worst-case regret of each rule over $(length(regimes)) regimes")
    for name in ("native", "fixed", "strip")
        Printf.@printf("  always %-7s %.3f\n", name, worst(_ -> name))
    end
    for other in ("strip", "fixed")
        scored = [(worst(reg -> reg[2] <= f && reg[1][3] >= b ? other : "native"), f, b)
                  for f in sort(unique(r[2] for r in rows)), b in (4, 8, 16, 32, 64)]
        for (w, f, b) in sort(vec(scored))[1:5]
            Printf.@printf("  %-5s iff fraction <= %.4f and B >= %-2d, else native: %.3f\n", other, f, b, w)
        end
    end
end

const SECTIONS = Dict("native1d" => native1d, "native1d_small" => io -> native1d(io; points = (5_000, 10_000), batch = false),
                      "fixed1d" => fixed1d, "fixed1d_large" => io -> fixed1d(io; points = (12_000,), slices = (4, 16)),
                      "carveout1d" => carveout1d, "native2d" => native2d,
                      "native2d_small" => io -> native2d(io; points = (5_000, 10_000), batches = ()),
                      "fixed2d" => io -> native2d(io; points = (), batches = ((5_000, 4), (5_000, 16), (5_000, 64), (12_000, 16)),
                                                  shapes = ((16, 8), (32, 16), (64, 64), (128, 192)),
                                                  rmaxes = (0.05, 0.1, 0.2, 0.3, 0.6, 1.5)),
                      "lag" => lag,
                      "harmonic" => harmonic, "sp2d_portable" => sp2d_portable, "batch1d" => batch1d)

section = get(ARGS, 1, "")
haskey(SECTIONS, section) || error("sections: $(join(sort(collect(keys(SECTIONS))), ", ")); got \"$section\"")
println("device=", CUDA.name(CUDA.device()), " section=", section, " max SM clock=",
        CUDA.NVML.max_clock_info(NVDEV).sm, " MHz")
open(SECTIONS[section], get(ARGS, 2, "launch_plans_$section.csv"), "w")
println("\nDONE_LAUNCH_PLANS")
