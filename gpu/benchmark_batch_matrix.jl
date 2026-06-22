# Production batch matrix benchmark (rows 1–7).
#
# Default is FAST (~1–2 min on GPU). Do not run full N=20k B=8064 in a dev loop.
#
#   julia --project=gpu gpu/benchmark_batch_matrix.jl                    # N=512 B=16
#   PROFILE=scaled julia --project=gpu gpu/benchmark_batch_matrix.jl     # N=4096 B=512
#   PROFILE=reference ALLOW_SLOW=1 julia --project=gpu gpu/benchmark_batch_matrix.jl
#       # N=20k B=8064 batch rows only; slice baseline = 8-sample extrapolation
#
# Full exact B×slice baseline: PROFILE=reference_full ALLOW_SLOW=1 (hours — acceptance only)
using Printf: @printf
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Random: Random
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

const _PROFILE_SIZES = Dict(
    :quick => (N = 512, B = 16),
    :scaled => (N = 4096, B = 512),
    :reference => (N = 20_000, B = 8064),
    :reference_full => (N = 20_000, B = 8064),
)

function _resolve_profile()
    raw = lowercase(strip(get(ENV, "PROFILE", get(ENV, "FAST", "1") == "1" ? "quick" : "scaled")))
    sym = Symbol(raw)
    sym in keys(_PROFILE_SIZES) || error(
        "PROFILE must be quick, scaled, reference, or reference_full; got $(repr(raw))",
    )
    return sym
end

function _resolve_sizes(profile::Symbol)
    if haskey(ENV, "BATCH_N") || haskey(ENV, "BATCH_B")
        return parse(Int, get(ENV, "BATCH_N", "512")), parse(Int, get(ENV, "BATCH_B", "16"))
    end
    sz = _PROFILE_SIZES[profile]
    return sz.N, sz.B
end

function _resolve_gpu_backend()
    if CUDA.functional()
        be = CUDA.CUDABackend()
        println("backend: CUDA ($(CUDA.name(CUDA.device())))")
        return SFC.GPUBackend(be), be
    end
    println("backend: KA.CPU (CUDA not functional)")
    return SFC.GPUBackend(KA.CPU()), KA.CPU()
end

function _bench_row(label::String, f::Function, ka_backend; warmup::Int = 1)
    print("  $label ... ")
    flush(stdout)
    for _ in 1:warmup
        f()
    end
    gpu_sync!(ka_backend)
    t0 = time()
    f()
    gpu_sync!(ka_backend)
    elapsed = time() - t0
    @printf("%.3f s\n", elapsed)
    return elapsed
end

"""Sampled slice baseline: time `n_sample` slices, extrapolate to full B."""
function _bench_slice_sampled(label::String, B::Int, n_sample::Int, f_slice::Function, ka_backend)
    n_sample = min(n_sample, B)
    idx = round.(Int, range(1, B; length = n_sample))
    t_sample = _bench_row(
        "$label (sample $n_sample / B=$B)",
        () -> begin
            for b in idx
                f_slice(b)
            end
        end,
        ka_backend,
    )
    t_extrap = t_sample * (B / n_sample)
    @printf("    extrapolated full B: %.1f s (%.1f min)\n", t_extrap, t_extrap / 60)
    return t_sample, t_extrap
end

function main()
    profile = _resolve_profile()
    N, B = _resolve_sizes(profile)
    large = N >= 4096 || B >= 512
    if large && get(ENV, "ALLOW_SLOW", "0") != "1"
        error(
            "Refusing large benchmark (N=$N B=$B profile=$profile) without ALLOW_SLOW=1. " *
            "Use PROFILE=quick for dev (~1–2 min), or PROFILE=scaled (~5 min).",
        )
    end

    slice_mode = get(ENV, "SLICE_BASELINE", profile === :reference_full ? "full" : profile === :reference ? "sample" : "off")
    prod_sample = parse(Int, get(ENV, "PROD_SAMPLE", profile === :reference ? "8" : "0"))
    warmup = parse(Int, get(ENV, "BATCH_WARMUP", "1"))
    n_dist = parse(Int, get(ENV, "N_DIST", "16"))
    n_val = parse(Int, get(ENV, "N_VAL", "8"))

    Random.seed!(1)
    x_fix = rand(Float32, 2, N)
    u_fix = rand(Float32, 2, N, B)
    x_var = rand(Float32, 2, N, B)
    u_var = rand(Float32, 2, N, B)
    edges = LinearBinEdges(collect(range(0.0f0, 2.0f0; length = n_dist + 1)))
    val_edges = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = n_val + 1)))
    sf = SFT.L2SFType()
    gpu_be, ka_backend = _resolve_gpu_backend()
    NB = length(edges.edges) - 1
    nv = length(val_edges.edges) - 1

    n_strips = cld(B, 16)
    n_tile_blocks = cld(N, 128) * (cld(N, 128) + 1) ÷ 2
    @printf(
        "batch matrix PROFILE=%s N=%d B=%d n_dist=%d n_val=%d strips(1D)≈%d strips(SP1D)≈%d tile_blocks≈%d\n",
        profile, N, B, n_dist, n_val, cld(B, 16), cld(B, 8), n_tile_blocks,
    )
    println("cases: individual-1D-fixed | individual-1D-varying | SP1D-fixed | SP1D-varying | SP2D-fixed | SP2D-varying | joint2D-fixed")

    _bench_row("individual 1D fixed-x (GPU batch)", () -> begin
        SFC.calculate_structure_function(
            sf, x_fix, u_fix, edges;
            backend = gpu_be, return_sums_and_counts = true, verbose = false,
        )
    end, ka_backend; warmup = warmup)

    if slice_mode == "full"
        _bench_row("individual 1D fixed-x slice baseline (B×serial)", () -> begin
            s = zeros(Float32, NB, B); c = zeros(UInt32, NB, B)
            for b in 1:B
                SFC.serial_calculate_structure_function!(
                    @view(s[:, b]), @view(c[:, b]), sf, x_fix, u_fix[:, :, b], edges;
                    verbose = false, show_progress = false,
                )
            end
        end, ka_backend; warmup = warmup)
    elseif slice_mode == "sample" && prod_sample > 0
        _bench_slice_sampled(
            "individual 1D fixed-x slice baseline", B, prod_sample,
            b -> begin
                s = zeros(Float32, NB); c = zeros(UInt32, NB)
                SFC.serial_calculate_structure_function!(
                    s, c, sf, x_fix, u_fix[:, :, b], edges;
                    verbose = false, show_progress = false,
                )
            end,
            ka_backend,
        )
    end

    _bench_row("individual 1D varying-x (GPU slices)", () -> begin
        s = zeros(Float32, NB, B); c = zeros(UInt32, NB, B)
        SFC.calculate_structure_function_slices!(s, c, sf, x_var, u_var, edges; backend = gpu_be)
    end, ka_backend; warmup = warmup)

    _bench_row("SP1D six-type fixed-x (GPU batch)", () -> begin
        SFC.calculate_structure_functions_single_pass(x_fix, u_fix, edges; backend = gpu_be)
    end, ka_backend; warmup = warmup)

    _bench_row("SP1D six-type varying-x (GPU slices)", () -> begin
        s = zeros(Float32, 6, NB, B); c = zeros(UInt32, 6, NB, B)
        SFC.calculate_structure_functions_single_pass_slices!(s, c, x_var, u_var, edges; backend = gpu_be)
    end, ka_backend; warmup = warmup)

    _bench_row("SP2D six-type fixed-x (GPU batch)", () -> begin
        SFC.calculate_structure_functions_single_pass_2d(x_fix, u_fix, edges, val_edges; backend = gpu_be)
    end, ka_backend; warmup = warmup)

    _bench_row("SP2D six-type varying-x (GPU slices)", () -> begin
        s = zeros(Float32, 6, NB, nv, B); c = zeros(UInt32, 6, NB, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_slices!(
            s, c, x_var, u_var, edges, val_edges; backend = gpu_be,
        )
    end, ka_backend; warmup = warmup)

    _bench_row("joint 2D single-type fixed-x (GPU batch)", () -> begin
        SFC.calculate_structure_function(
            sf, x_fix, u_fix, edges, val_edges;
            backend = gpu_be, return_sums_and_counts = true, verbose = false,
        )
    end, ka_backend; warmup = warmup)

    println("done.")
end

main()
