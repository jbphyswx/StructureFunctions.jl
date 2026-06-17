#!/usr/bin/env julia
"""
    prove_f32_accumulation.jl

Validate references, then compare Float32 accumulation orders.

**Run in SLURM** with `ENV["N"] = "20000"` for full proof.

    include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
    ENV["N"] = "20000"
    include_gpu("prove_f32_accumulation.jl")
"""

using Printf: @printf
using Random: Random
using StaticArrays: StaticArrays as SA
using StructureFunctions: HelperFunctions as SFH, StructureFunctionTypes as SFT, LinearBinEdges

const _LOGIN_N_MAX = 8_000
const _WORKGROUP_SIZE = 256

function _check_n_budget!(N::Int)
    in_slurm = haskey(ENV, "SLURM_JOB_ID")
    allowed = get(ENV, "ALLOW_LARGE_N", "0") == "1"
    N > _LOGIN_N_MAX && !in_slurm && !allowed &&
        error("N=$N too heavy for login node; run in SLURM or set ALLOW_LARGE_N=1.")
    return nothing
end

include(joinpath(@__DIR__, "GPUPrototypeKernels.jl"))

"""Sum bin `target_bin` in stride vs private **Float32** order; count no-op stride adds."""
function sum_bin_f32_orders(
    x_mat, u_mat, sft, bin_edges, target_bin::Int, nworkers::Int,
)
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    total_pairs = N_points * (N_points - 1) ÷ 2
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val

    stride_s = zero(Float32)
    private_s = zero(Float32)
    noop = 0
    n = 0

    for worker in 1:nworkers
        partial = zero(Float32)
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            X1 = SA.SVector{3, Float32}(x3[1, i], x3[2, i], x3[3, i])
            X2 = SA.SVector{3, Float32}(x3[1, j], x3[2, j], x3[3, j])
            dX = X2 - X1
            dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
            bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
            if bin == target_bin
                U1 = SA.SVector{3, Float32}(u3[1, i], u3[2, i], u3[3, i])
                U2 = SA.SVector{3, Float32}(u3[1, j], u3[2, j], u3[3, j])
                v = sft(U2 - U1, SFH.r̂(X1, X2))
                n += 1
                s2 = stride_s + v
                noop += s2 == stride_s
                stride_s = s2
                partial += v
            end
            k += nworkers
        end
        private_s += partial
    end
    return stride_s, private_s, noop, n
end

function print_f64_crosscheck(v, target_bin::Int)
    b = target_bin
    ref_b = v.serial.sums[b]
    @printf("max|f64_serial - f64_private|    (all bins) = %.6g\n", v.max_serial_private)
    @printf("max|f64_serial - f64_stride|     (all bins) = %.6g\n", v.max_serial_stride)
    @printf("max|f64_serial - f64_blockshared| (all bins) = %.6g\n", v.max_serial_blockshared)
    @printf("bin[%d] f64_serial     = %.6g\n", b, ref_b)
    @printf("bin[%d] f64_private    = %.6g  (Δ serial %+.6g)\n",
        b, v.private.sums[b], v.private.sums[b] - ref_b)
    @printf("bin[%d] f64_stride     = %.6g  (Δ serial %+.6g)\n",
        b, v.stride.sums[b], v.stride.sums[b] - ref_b)
    @printf("bin[%d] f64_blockshared = %.6g  (Δ serial %+.6g)\n",
        b, v.blockshared.sums[b], v.blockshared.sums[b] - ref_b)
    @printf("bin[%d] BigFloat        = %.6g  (Δ serial %+.6g)\n",
        b, v.bigfloat.sum, v.bf_diff)
    pct_f32 = ref_b == 0 ? NaN : 100 * v.f32_vs_f64_bin / ref_b
    @printf("bin[%d] f32_serial      = %.6g  (Δ serial %+.6g, %.3f%%) [diagnostic, not ref]\n",
        b, v.f32_serial_bin, v.f32_vs_f64_bin, pct_f32)
    return nothing
end

"""Compare each Float32 path to its **same-schedule Float64** twin."""
function print_f32_vs_f64_twin(target_bin::Int, pairs...)
    b = target_bin
    @printf("\n--- bin[%d] Float32 vs same-schedule Float64 twin ---\n", b)
    for (label, f32_res, f64_sums) in pairs
        got = Float64(f32_res.sums[b])
        ref = f64_sums[b]
        Δ = got - ref
        pct = ref == 0 ? NaN : 100 * Δ / ref
        @printf("bin[%d] %-16s f32=%.6g  f64_twin=%.6g  (Δ %+.6g, %.3f%%)\n",
            b, label, got, ref, Δ, pct)
    end
    return nothing
end

function main()
    N = parse(Int, get(ENV, "N", "500"))
    _check_n_budget!(N)
    target_bin = parse(Int, get(ENV, "TARGET_BIN", "5"))
    Random.seed!(42)
    FT = Float32
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    nworkers = min(262_144, N * (N - 1) ÷ 2)

    println("=== Step 1: validate Float64 references (all schedules) ===")
    println("N = $N  target_bin = $target_bin  Threads.nthreads() = $(Threads.nthreads())")
    v = validate_f64_references(
        x_cpu, u_cpu, sft, bin_edges;
        nworkers = nworkers,
        workgroup_size = _WORKGROUP_SIZE,
        target_bin = target_bin,
    )
    print_f64_crosscheck(v, target_bin)
    if v.reference_ok
        println("PASS: serial/private/stride/blockshared f64 all agree; BigFloat matches bin.")
    else
        println("FAIL: Float64 reference NOT validated — stop.")
        @printf("  f64 schedules ok = %s\n", v.f64_ok)
        @printf("  BigFloat ok      = %s\n", v.bf_ok)
        return v
    end

    ref = v.serial.sums
    f64_private = v.private.sums
    f64_stride = v.stride.sums
    f64_blockshared = v.blockshared.sums

    println("\n=== Step 2: Float32 paths vs same-schedule f64 twins ===")
    stride = cpu_stride_global_histogram(x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers)
    private = cpu_private_histogram(x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers)
    bs = cpu_blockshared_histogram(
        x_cpu, u_cpu, sft, bin_edges;
        nworkers = nworkers, workgroup_size = _WORKGROUP_SIZE,
    )

    @printf("max|f64_serial - stride_f32|      = %.6g\n", maximum(abs.(ref .- stride.sums)))
    @printf("max|f64_serial - private_f32|    = %.6g\n", maximum(abs.(ref .- private.sums)))
    @printf("max|f64_serial - blockshared_f32| = %.6g\n", maximum(abs.(ref .- bs.sums)))
    @printf("max|f64_stride_twin - stride_f32|      = %.6g\n",
        maximum(abs.(f64_stride .- stride.sums)))
    @printf("max|f64_private_twin - private_f32|    = %.6g\n",
        maximum(abs.(f64_private .- private.sums)))
    @printf("max|f64_blockshared_twin - blockshared_f32| = %.6g\n",
        maximum(abs.(f64_blockshared .- bs.sums)))

    print_f32_vs_f64_twin(
        target_bin,
        ("stride", stride, f64_stride),
        ("private", private, f64_private),
        ("blockshared", bs, f64_blockshared),
    )

    println("\n=== Step 3: bin-level f32 replay (kernel accumulation order) ===")
    stride_r, private_r, noop, npairs = sum_bin_f32_orders(
        x_cpu, u_cpu, sft, bin_edges, target_bin, nworkers,
    )
    @printf("bin[%d] pairs = %d\n", target_bin, npairs)
    @printf("f64_serial bin sum             = %.6g\n", ref[target_bin])
    @printf("f64_stride twin bin sum        = %.6g\n", f64_stride[target_bin])
    @printf("stride-order f32 replay          = %.6g  (Δ f64_stride_twin %+.6g)\n",
        stride_r, stride_r - f64_stride[target_bin])
    @printf("private-order f32 replay         = %.6g  (Δ f64_private_twin %+.6g)\n",
        private_r, private_r - f64_private[target_bin])
    @printf("no-op stride f32 adds            = %d / %d (%.3f%%)\n",
        noop, npairs, 100 * noop / max(npairs, 1))
    @printf("replay stride vs CPU stride      = %.6g\n", stride_r - stride.sums[target_bin])
    @printf("replay private vs CPU private    = %.6g\n", private_r - private.sums[target_bin])

    return v
end

main()
