# The choice of a native launch plan: a formula over the device and the call names the plans worth trying, and the
# second call of each class of calls times them once and keeps the fastest.

"""The native plans of one call's types on one device: `candidates(N, B, fixed, evaluations, share)` gives the plans
worth trying for a call, in order of preference. A call of at least `choose_from` pair evaluations samples its in-range
share; the first call of its class ([`_cuda_call_class`](@ref)) takes the first candidate, the class joining `seen`, and
the second times the candidates, `chosen` keeping the fastest. A smaller call takes the first candidate at a share of
one, `full` keeping it by `(fixed, N, B)`."""
struct CUDAChoice{C}
    candidates::C
    choose_from::Float64
    chosen::Dict{NTuple{4, Int}, Any}
    seen::Set{NTuple{4, Int}}
    full::Dict{NTuple{3, Int}, Any}
    lock::ReentrantLock
end

CUDAChoice(candidates, choose_from::Real) =
    CUDAChoice(candidates, Float64(choose_from), Dict{NTuple{4, Int}, Any}(), Set{NTuple{4, Int}}(),
               Dict{NTuple{3, Int}, Any}(), ReentrantLock())

"""The class of a call whose plans are measured once: a batch over shared positions or not, the pair evaluations to
a factor of 4, the in-range share's band and the strip width its slices allow."""
function _cuda_call_class(fixed::Bool, B::Int, evaluations::Real, share::Real)
    band = share < 0.02 ? 0 : share < 0.2 ? 1 : share < 0.6 ? 2 : 3
    return (Int(fixed), floor(Int, log(4, max(evaluations, 1))), band, fixed ? min(prevpow(2, B), 8) : 1)
end

"""The share of the tile pairs of `N` points at `tile` that the cull memo `cull` keeps."""
function _cuda_kept(cull, N::Int, tile::Int)
    n = cld(N, tile)
    return SFC.n_pair_blocks(SFC.schedule_for(cull, N, tile)) / (n * (n + 1) ÷ 2)
end

"""Tile pairs of `N` points at `tile`."""
_cuda_blocks(N::Int, tile::Int) = (n = cld(N, tile); n * (n + 1) ÷ 2)

"""The plan of `choice` for a launch over `N` points and `B` slices (`fixed`: a batch over shared positions) of the
kernel coordinates `x` binned by `ddig` into `NB` distance bins: the first candidate at a share of one when the pair
evaluations it makes (with the cull memo `cull`, their share in the tile pairs it keeps) are fewer than
`choice.choose_from`; else, with the in-range share sampled over its schedule, the first candidate on the class's first
call, and from its second call the class's measured plan, measured then by timing each candidate through
`launch!(plan, sums, counts)` into scratch buffers like `out` and `cnt`."""
function _cuda_plan(launch!::L, choice::CUDAChoice, out, cnt, x, ddig, NB::Int, N::Int, B::Int, fixed::Bool, geom,
                    cull) where {L}
    all_pairs = (N * (N - 1) ÷ 2) * B
    full = lock(choice.lock) do
        get!(() -> first(choice.candidates(N, B, fixed, all_pairs, 1.0)), choice.full, (Int(fixed), N, B))
    end
    t = _cuda_tile(full)
    evaluations = cull === nothing ? all_pairs : round(Int, all_pairs * _cuda_kept(cull, N, t))
    evaluations < choice.choose_from && return full
    share = SFC.gpu_in_range_fraction(CUDA.CUDABackend(), x, ddig, NB, geom, cull, t)
    class = _cuda_call_class(fixed, B, evaluations, share)
    return lock(choice.lock) do
        haskey(choice.chosen, class) && return choice.chosen[class]
        candidates = choice.candidates(N, B, fixed, evaluations, share)
        length(candidates) == 1 && return (choice.chosen[class] = first(candidates))
        class in choice.seen || (push!(choice.seen, class); return first(candidates))
        return (choice.chosen[class] = _cuda_fastest(launch!, candidates, out, cnt))
    end
end

"""Timed rounds over the candidates of a class, and the ratio to the fastest past which a candidate leaves them."""
const CU_TIMING_ROUNDS = 3
const CU_TIMING_KEEP = 1.5

"""The candidate whose launch into zeroed scratch buffers like `out` and `cnt` takes the least time: every candidate is
compiled and launched, then launched once more after the last has compiled, then timed in [`CU_TIMING_ROUNDS`](@ref)
interleaved rounds, each keeping its best time; from the second round on, the candidates slower than
[`CU_TIMING_KEEP`](@ref) times the fastest leave the rounds. `CUDA.@elapsed` includes host time inside a launch."""
function _cuda_fastest(launch!, candidates, out, cnt)
    s, c = fill!(similar(out), 0), fill!(similar(cnt), 0)
    foreach(plan -> launch!(plan, s, c), candidates)
    foreach(plan -> launch!(plan, s, c), candidates)
    best = fill(Inf32, length(candidates))
    live = collect(eachindex(candidates))
    for round in 1:CU_TIMING_ROUNDS
        for k in live
            best[k] = min(best[k], CUDA.@elapsed launch!(candidates[k], s, c))
        end
        round >= 2 && (live = filter(k -> best[k] <= CU_TIMING_KEEP * minimum(best), live))
    end
    return candidates[argmin(best)]
end
