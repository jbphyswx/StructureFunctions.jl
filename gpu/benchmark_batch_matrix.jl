# Production auxiliary-axis batch matrix benchmark.
#
# Preferred REPL use inside a SLURM GPU allocation:
#
#   include("gpu/benchmark_batch_matrix.jl")
#   run_batch_matrix_benchmark(profile = :quick)
#   run_batch_matrix_benchmark(profile = :reference, allow_slow = true,
#       cases = (:individual_fixed, :individual_fixed_gpu_sample, :individual_varying))
#
# CLI compatibility remains available:
#
#   julia --project=gpu gpu/benchmark_batch_matrix.jl
#   PROFILE=reference ALLOW_SLOW=1 julia --project=gpu gpu/benchmark_batch_matrix.jl
#
# The explicit slice baselines in this file are GPU baselines. CPU serial baselines
# belong in CPU benchmark scripts and are intentionally not mixed into this matrix.
using Printf: @printf
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Random: Random
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

const PROFILE_SIZES = Dict(
    :quick => (N = 512, B = 16),
    :scaled => (N = 4096, B = 512),
    :reference => (N = 20_000, B = 8064),
)

const DEFAULT_CASES = (
    :individual_fixed,
    :individual_fixed_gpu_sample,
    :individual_varying,
    :sp1d_fixed,
    :sp1d_varying_gpu_sample,
    :sp2d_fixed,
    :sp2d_varying_gpu_sample,
    :joint2d_fixed,
)

const ALL_CASES = (
    :individual_fixed,
    :individual_fixed_gpu_sample,
    :individual_fixed_gpu_full,
    :individual_varying,
    :individual_varying_explicit_gpu_sample,
    :sp1d_fixed,
    :sp1d_varying_gpu_sample,
    :sp1d_varying_explicit_gpu_sample,
    :sp1d_varying,
    :sp2d_fixed,
    :sp2d_varying_gpu_sample,
    :sp2d_varying_explicit_gpu_sample,
    :sp2d_varying,
    :joint2d_fixed,
)

function _profile_size(profile::Symbol)
    haskey(PROFILE_SIZES, profile) ||
        error("profile must be one of $(collect(keys(PROFILE_SIZES))); got $profile")
    return PROFILE_SIZES[profile]
end

function _resolve_gpu_backend(backend)
    if backend !== nothing
        backend isa SFC.GPUBackend && return backend, backend.backend
        return SFC.GPUBackend(backend), backend
    end
    if CUDA.functional()
        ka_backend = CUDA.CUDABackend()
        println("backend: CUDA ($(CUDA.name(CUDA.device())))")
        return SFC.GPUBackend(ka_backend), ka_backend
    end
    ka_backend = KA.CPU()
    println("backend: KA.CPU (CUDA not functional)")
    return SFC.GPUBackend(ka_backend), ka_backend
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

function _stage_for_backend(ka_backend, a)
    ka_backend isa CUDA.CUDABackend && return CUDA.CuArray(a)
    return a
end

function _sample_indices(B::Int, n_sample::Int)
    n_sample = clamp(n_sample, 1, B)
    n_sample == 1 && return [1]
    return unique(round.(Int, range(1, B; length = n_sample)))
end

function _sample_batch(a, idx)
    return @views copy(a[:, :, idx])
end

function _bench_explicit_gpu_shared_loop(
    label::String,
    sf,
    ka_backend,
    x,
    u,
    edges,
    sample_indices;
    warmup::Int,
    extrapolate::Bool,
)
    xd = _stage_for_backend(ka_backend, x)
    ud = _stage_for_backend(ka_backend, u)
    B = size(u, 3)
    n_sample = length(sample_indices)
    elapsed = _bench_row(
        label,
        () -> begin
            @views for b in sample_indices
                SFC.gpu_calculate_structure_function(
                    sf, ka_backend, xd, ud[:, :, b], edges;
                    return_sums_and_counts = true,
                )
            end
        end,
        ka_backend;
        warmup = warmup,
    )
    if extrapolate
        full_estimate = elapsed * B / n_sample
        @printf("    explicit GPU loop estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
    end
    return elapsed
end

function _bench_explicit_gpu_shared_loop_full(
    sf,
    ka_backend,
    x,
    u,
    edges;
    warmup::Int,
)
    return _bench_explicit_gpu_shared_loop(
        "individual 1D fixed-x explicit GPU slice loop (full B)",
        sf,
        ka_backend,
        x,
        u,
        edges,
        axes(u, 3);
        warmup = warmup,
        extrapolate = false,
    )
end

function _bench_explicit_gpu_varying_sf_loop(
    label::String,
    sf,
    ka_backend,
    x,
    u,
    edges,
    sample_indices;
    warmup::Int,
    extrapolate::Bool,
)
    xd = _stage_for_backend(ka_backend, x)
    ud = _stage_for_backend(ka_backend, u)
    B = size(u, 3)
    n_sample = length(sample_indices)
    elapsed = _bench_row(
        label,
        () -> begin
            @views for b in sample_indices
                SFC.gpu_calculate_structure_function(
                    sf, ka_backend, xd[:, :, b], ud[:, :, b], edges;
                    return_sums_and_counts = true,
                )
            end
        end,
        ka_backend;
        warmup = warmup,
    )
    if extrapolate
        full_estimate = elapsed * B / n_sample
        @printf("    explicit optimized GPU SF loop estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
    end
    return elapsed
end

function _bench_explicit_gpu_varying_sp1d_loop(
    label::String,
    gpu_backend,
    ka_backend,
    x,
    u,
    edges,
    sample_indices;
    warmup::Int,
    extrapolate::Bool,
)
    xd = _stage_for_backend(ka_backend, x)
    ud = _stage_for_backend(ka_backend, u)
    B = size(u, 3)
    n_sample = length(sample_indices)
    elapsed = _bench_row(
        label,
        () -> begin
            @views for b in sample_indices
                SFC.calculate_structure_functions_single_pass(
                    xd[:, :, b], ud[:, :, b], edges; backend = gpu_backend,
                )
            end
        end,
        ka_backend;
        warmup = warmup,
    )
    if extrapolate
        full_estimate = elapsed * B / n_sample
        @printf("    explicit optimized GPU SP1D loop estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
    end
    return elapsed
end

function _bench_explicit_gpu_varying_sp2d_loop(
    label::String,
    gpu_backend,
    ka_backend,
    x,
    u,
    edges,
    value_edges,
    sample_indices;
    warmup::Int,
    extrapolate::Bool,
)
    xd = _stage_for_backend(ka_backend, x)
    ud = _stage_for_backend(ka_backend, u)
    B = size(u, 3)
    n_sample = length(sample_indices)
    elapsed = _bench_row(
        label,
        () -> begin
            @views for b in sample_indices
                SFC.calculate_structure_functions_single_pass_2d(
                    xd[:, :, b], ud[:, :, b], edges, value_edges; backend = gpu_backend,
                )
            end
        end,
        ka_backend;
        warmup = warmup,
    )
    if extrapolate
        full_estimate = elapsed * B / n_sample
        @printf("    explicit optimized GPU SP2D loop estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
    end
    return elapsed
end

function _parse_cases(raw::AbstractString)
    isempty(strip(raw)) && return DEFAULT_CASES
    raw == "all" && return ALL_CASES
    return Tuple(Symbol(strip(part)) for part in split(raw, ",") if !isempty(strip(part)))
end

"""
    run_batch_matrix_benchmark(; kwargs...)

Run the maintained auxiliary-axis GPU benchmark matrix.

Keyword controls:

- `profile`: `:quick`, `:scaled`, or `:reference`.
- `N`, `B`: override the profile sizes.
- `n_dist`, `n_val`: distance/value bin counts.
- `cases`: tuple of case symbols. Use `ALL_CASES` for the full matrix.
- `explicit_samples`: number of slices for `:individual_fixed_gpu_sample`.
- `allow_slow`: required for large `N` or `B`.
- `backend`: optional KA backend, usually left as CUDA auto-detection.

The default reference cases intentionally exclude the slow varying-position SP1D/SP2D
full-B slice routes. They include sampled varying-position SP1D/SP2D routes with an
extrapolated full-B estimate. Add `:sp1d_varying` or `:sp2d_varying` explicitly only
when measuring the full route.
"""
function run_batch_matrix_benchmark(;
    profile::Symbol = :quick,
    N::Union{Nothing, Int} = nothing,
    B::Union{Nothing, Int} = nothing,
    n_dist::Int = 16,
    n_val::Int = 8,
    cases = DEFAULT_CASES,
    explicit_samples::Int = profile === :reference ? 8 : 4,
    warmup::Int = 1,
    allow_slow::Bool = false,
    backend = nothing,
    seed::Int = 1,
)
    sz = _profile_size(profile)
    N = something(N, sz.N)
    B = something(B, sz.B)
    large = N >= 4096 || B >= 512
    if large && !allow_slow
        error(
            "Refusing large benchmark (N=$N B=$B profile=$profile) without allow_slow=true. " *
            "Use profile=:quick for dev, or pass allow_slow=true deliberately.",
        )
    end
    unknown = setdiff(Symbol.(cases), ALL_CASES)
    isempty(unknown) || error("unknown benchmark cases: $unknown; valid cases are $(ALL_CASES)")

    Random.seed!(seed)
    x_fix = rand(Float32, 2, N)
    u_fix = rand(Float32, 2, N, B)
    x_var = rand(Float32, 2, N, B)
    u_var = rand(Float32, 2, N, B)
    edges = LinearBinEdges(collect(range(0.0f0, 2.0f0; length = n_dist + 1)))
    val_edges = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = n_val + 1)))
    sf = SFT.L2SFType()
    gpu_backend, ka_backend = _resolve_gpu_backend(backend)
    NB = length(edges.edges) - 1
    nv = length(val_edges.edges) - 1

    n_tile_blocks = cld(N, 128) * (cld(N, 128) + 1) ÷ 2
    @printf(
        "batch matrix profile=%s N=%d B=%d n_dist=%d n_val=%d strips(1D)=%d strips(SP1D)=%d tile_blocks=%d (NB must be <= 127)\n",
        profile, N, B, n_dist, n_val, cld(B, 16), cld(B, 8), n_tile_blocks,
    )
    println("cases: $(join(string.(cases), ", "))")
    rows = Pair{Symbol, Float64}[]

    if :individual_fixed in cases
        push!(rows, :individual_fixed => _bench_row("individual 1D fixed-x (GPU batch)", () -> begin
            SFC.calculate_structure_function(
                sf, x_fix, u_fix, edges;
                backend = gpu_backend, return_sums_and_counts = true, verbose = false,
            )
        end, ka_backend; warmup = warmup))
    end

    if :individual_fixed_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        label = "individual 1D fixed-x explicit GPU slice loop (sample $(length(idx)) / B=$B)"
        push!(rows, :individual_fixed_gpu_sample => _bench_explicit_gpu_shared_loop(
            label, sf, ka_backend, x_fix, u_fix, edges, idx;
            warmup = warmup,
            extrapolate = true,
        ))
    end

    if :individual_fixed_gpu_full in cases
        push!(rows, :individual_fixed_gpu_full => _bench_explicit_gpu_shared_loop_full(
            sf, ka_backend, x_fix, u_fix, edges; warmup = warmup,
        ))
    end

    if :individual_varying in cases
        push!(rows, :individual_varying => _bench_row("individual 1D varying-x (GPU slices)", () -> begin
            s = zeros(Float32, NB, B)
            c = zeros(UInt32, NB, B)
            SFC.calculate_structure_function_slices!(s, c, sf, x_var, u_var, edges; backend = gpu_backend)
        end, ka_backend; warmup = warmup))
    end

    if :individual_varying_explicit_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        label = "individual 1D varying-x explicit optimized GPU loop (sample $(length(idx)) / B=$B)"
        push!(rows, :individual_varying_explicit_gpu_sample => _bench_explicit_gpu_varying_sf_loop(
            label, sf, ka_backend, x_var, u_var, edges, idx;
            warmup = warmup,
            extrapolate = true,
        ))
    end

    if :sp1d_fixed in cases
        push!(rows, :sp1d_fixed => _bench_row("SP1D six-invariant fixed-x (GPU batch)", () -> begin
            SFC.calculate_structure_functions_single_pass(x_fix, u_fix, edges; backend = gpu_backend)
        end, ka_backend; warmup = warmup))
    end

    if :sp1d_varying_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        x_sample = _sample_batch(x_var, idx)
        u_sample = _sample_batch(u_var, idx)
        B_sample = length(idx)
        elapsed = _bench_row(
            "SP1D six-invariant varying-x (GPU slices sample $B_sample / B=$B)",
            () -> begin
                s = zeros(Float32, 6, NB, B_sample)
                c = zeros(UInt32, 6, NB, B_sample)
                SFC.calculate_structure_functions_single_pass_slices!(
                    s, c, x_sample, u_sample, edges; backend = gpu_backend,
                )
            end,
            ka_backend;
            warmup = warmup,
        )
        full_estimate = elapsed * B / B_sample
        @printf("    varying SP1D estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
        push!(rows, :sp1d_varying_gpu_sample => elapsed)
    end

    if :sp1d_varying_explicit_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        label = "SP1D varying-x explicit optimized GPU loop (sample $(length(idx)) / B=$B)"
        push!(rows, :sp1d_varying_explicit_gpu_sample => _bench_explicit_gpu_varying_sp1d_loop(
            label, gpu_backend, ka_backend, x_var, u_var, edges, idx;
            warmup = warmup,
            extrapolate = true,
        ))
    end

    if :sp1d_varying in cases
        push!(rows, :sp1d_varying => _bench_row("SP1D six-invariant varying-x (GPU slices full B)", () -> begin
            s = zeros(Float32, 6, NB, B)
            c = zeros(UInt32, 6, NB, B)
            SFC.calculate_structure_functions_single_pass_slices!(s, c, x_var, u_var, edges; backend = gpu_backend)
        end, ka_backend; warmup = warmup))
    end

    if :sp2d_fixed in cases
        push!(rows, :sp2d_fixed => _bench_row("SP2D six-invariant fixed-x (GPU batch)", () -> begin
            SFC.calculate_structure_functions_single_pass_2d(x_fix, u_fix, edges, val_edges; backend = gpu_backend)
        end, ka_backend; warmup = warmup))
    end

    if :sp2d_varying_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        x_sample = _sample_batch(x_var, idx)
        u_sample = _sample_batch(u_var, idx)
        B_sample = length(idx)
        elapsed = _bench_row(
            "SP2D six-invariant varying-x (GPU slices sample $B_sample / B=$B)",
            () -> begin
                s = zeros(Float32, 6, NB, nv, B_sample)
                c = zeros(UInt32, 6, NB, nv, B_sample)
                SFC.calculate_structure_functions_single_pass_2d_slices!(
                    s, c, x_sample, u_sample, edges, val_edges; backend = gpu_backend,
                )
            end,
            ka_backend;
            warmup = warmup,
        )
        full_estimate = elapsed * B / B_sample
        @printf("    varying SP2D estimate for full B: %.1f s (%.1f min)\n",
            full_estimate, full_estimate / 60)
        push!(rows, :sp2d_varying_gpu_sample => elapsed)
    end

    if :sp2d_varying_explicit_gpu_sample in cases
        idx = _sample_indices(B, explicit_samples)
        label = "SP2D varying-x explicit optimized GPU loop (sample $(length(idx)) / B=$B)"
        push!(rows, :sp2d_varying_explicit_gpu_sample => _bench_explicit_gpu_varying_sp2d_loop(
            label, gpu_backend, ka_backend, x_var, u_var, edges, val_edges, idx;
            warmup = warmup,
            extrapolate = true,
        ))
    end

    if :sp2d_varying in cases
        push!(rows, :sp2d_varying => _bench_row("SP2D six-invariant varying-x (GPU slices full B)", () -> begin
            s = zeros(Float32, 6, NB, nv, B)
            c = zeros(UInt32, 6, NB, nv, B)
            SFC.calculate_structure_functions_single_pass_2d_slices!(
                s, c, x_var, u_var, edges, val_edges; backend = gpu_backend,
            )
        end, ka_backend; warmup = warmup))
    end

    if :joint2d_fixed in cases
        push!(rows, :joint2d_fixed => _bench_row("joint 2D single-type fixed-x (GPU batch)", () -> begin
            SFC.calculate_structure_function(
                sf, x_fix, u_fix, edges, val_edges;
                backend = gpu_backend, return_sums_and_counts = true, verbose = false,
            )
        end, ka_backend; warmup = warmup))
    end

    println("done.")
    return (; profile, N, B, n_dist, n_val, cases = Tuple(cases), rows = Dict(rows))
end

function main()
    profile = Symbol(lowercase(strip(get(ENV, "PROFILE", "quick"))))
    sizes = _profile_size(profile)
    N = parse(Int, get(ENV, "BATCH_N", string(sizes.N)))
    B = parse(Int, get(ENV, "BATCH_B", string(sizes.B)))
    cases = _parse_cases(get(ENV, "CASES", join(string.(DEFAULT_CASES), ",")))
    return run_batch_matrix_benchmark(;
        profile,
        N,
        B,
        n_dist = parse(Int, get(ENV, "N_DIST", "16")),
        n_val = parse(Int, get(ENV, "N_VAL", "8")),
        cases,
        explicit_samples = parse(Int, get(ENV, "EXPLICIT_SAMPLES", profile === :reference ? "8" : "4")),
        warmup = parse(Int, get(ENV, "BATCH_WARMUP", "1")),
        allow_slow = get(ENV, "ALLOW_SLOW", "0") == "1",
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
