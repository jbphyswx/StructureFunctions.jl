#!/usr/bin/env julia
# Stage 0 accuracy harness: measured relative error of Float32 accumulation orderings against a
# Float64 reference, on the real per-pair values of the 1D pair loop.
#
# This is the measurement behind the accumulator-eltype decision: the accumulator stays Float32
# (widening halves the GPU shared-memory histogram and with it occupancy — the measured bottleneck),
# and accuracy comes from the accumulation STRUCTURE. Blocked/pairwise summation reduces roundoff
# growth from O(n) to O(log n) at no bandwidth cost; this prints how much that buys at each N.
#
# Usage: julia --project=benchmark benchmark/accuracy.jl

using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using Printf, Random

const SFTYPE = SFT.L2SFType()

# Per-pair L2SF values of the pairs of one bin, exactly as the kernel computes them (Float32 path),
# plus the Float64-computed reference values of the same pairs.
function bin_values(N)
    Random.seed!(7)
    x64 = rand(Float64, 2, N); u64 = randn(Float64, 2, N)
    x32 = Float32.(x64); u32 = Float32.(u64)
    v32 = Float32[]; v64 = Float64[]
    for i in 1:(N - 1), j in (i + 1):N
        dx64 = (x64[1, j] - x64[1, i], x64[2, j] - x64[2, i])
        r64 = sqrt(dx64[1]^2 + dx64[2]^2)
        du64 = (u64[1, j] - u64[1, i], u64[2, j] - u64[2, i])
        push!(v64, ((du64[1] * dx64[1] + du64[2] * dx64[2]) / r64)^2)

        dx32 = (x32[1, j] - x32[1, i], x32[2, j] - x32[2, i])
        r32 = sqrt(dx32[1]^2 + dx32[2]^2)
        du32 = (u32[1, j] - u32[1, i], u32[2, j] - u32[2, i])
        push!(v32, ((du32[1] * dx32[1] + du32[2] * dx32[2]) / r32)^2)
    end
    return v32, v64
end

naive_sum(v::Vector{T}) where {T} = (s = zero(T); for x in v; s += x; end; s)

# Panel-blocked naive sum: partials of `blk` values summed naively, then partials summed naively.
# This is the ordering a j-panel kernel produces for free.
function blocked_sum(v::Vector{T}, blk::Int) where {T}
    s = zero(T)
    i = 1
    while i <= length(v)
        hi = min(i + blk - 1, length(v))
        p = zero(T)
        for k in i:hi
            p += v[k]
        end
        s += p
        i = hi + 1
    end
    return s
end

function main()
    @printf("%s\nFloat32 accumulation error vs Float64 reference (same pairs, kernel-order values)\n%s\n",
            "="^100, "="^100)
    @printf("%8s %12s | %14s %14s %14s %14s\n",
            "N", "pairs", "naive f32", "blk=512 f32", "blk=2048 f32", "pairwise f32")
    for N in (1_000, 3_000, 10_000)
        v32, v64 = bin_values(N)
        ref = sum(v64)                                  # Float64 pairwise (Base) = reference
        rel(x) = abs(Float64(x) - ref) / abs(ref)
        @printf("%8d %12d | %14.3e %14.3e %14.3e %14.3e\n",
                N, length(v32),
                rel(naive_sum(v32)),
                rel(blocked_sum(v32, 512)),
                rel(blocked_sum(v32, 2048)),
                rel(sum(v32)))                          # Base pairwise on Float32
    end
    println("""
naive f32     = today's kernel ordering (straight left-to-right +=)
blk f32       = j-panel partial sums (the Stage-3 blocked kernel's free ordering)
pairwise f32  = Base's pairwise reduction (log-depth; the achievable floor without widening)
The per-value input error floor for Float32 data is ~1e-7 regardless of summation order.""")
end

main()
