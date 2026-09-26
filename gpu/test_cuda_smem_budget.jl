# Static shared memory of every tiled kernel family on CUDA, as `ptxas` reports it: each kernel a call
# compiles is compared with the byte function its launcher decides by. Portable kernels are launched
# through their launcher's portable half, at the widest configuration the byte function admits and one
# past it; native kernels through the public entry, at the plan the native plan function returns. Every
# call agrees with the serial answer.
using CUDA, Random, Printf
using Logging: Logging
using Test: Test
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB

const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const CE = Base.get_extension(SF, :StructureFunctionsCUDAExt)
const BE = CUDA.CUDABackend()
const DEV = CB.GPUBackend(BE)
const SER = CB.SerialBackend()
const CAPS = SFC.gpu_device_caps(BE)
const BUDGET = SFC.gpu_static_smem_budget(CAPS)
const OP = SFT.L2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts
const N = 300
const B = 3
const F64 = Float64

failures = String[]

function check(name, ok::Bool)
    ok || push!(failures, name)
    @printf("%-86s %s\n", name, ok ? "ok" : "FAILED")
    return ok
end

"""The call's result and the static shared bytes `ptxas` reports for each kernel entry it compiled."""
function compiled_smem(f)
    logger = Test.TestLogger(; min_level = Logging.Debug)
    result = Logging.with_logger(f, logger)
    CUDA.synchronize()
    smem = Dict{String, Int}()
    for rec in logger.logs
        msg = string(rec.message)
        occursin("PTX compiler log", msg) || continue
        entry = nothing
        for line in split(msg, '\n')
            m = match(r"Compiling entry function '([^']+)'", line)
            if m !== nothing
                entry = String(m[1])
                smem[entry] = 0
                continue
            end
            m = match(r"(\d+) bytes smem", line)
            m === nothing || entry === nothing || (smem[entry] = parse(Int, m[1]))
        end
    end
    return result, smem
end

"""The reported bytes of the one compiled entry whose name contains `pattern`, or `nothing`."""
function entry_bytes(smem, pattern)
    hits = [k for k in keys(smem) if occursin(pattern, k)]
    return length(hits) == 1 ? smem[only(hits)] : nothing
end

"""Sums and counts of a portable launch, shaped as the serial result they are compared with."""
struct Raw
    sums
    counts
end

parts(r::NamedTuple) = reduce(vcat, [parts(v) for v in values(r)])
parts(r::SFO.HelmholtzDecomposition2D) = Any[Array(r.rotational_sums), Array(r.rotational_counts),
                                             Array(r.divergent_sums), Array(r.divergent_counts)]
parts(r) = Any[Array(r.sums), Array(r.counts)]

"""Whether every array of `got` equals `ref`'s: exactly for integer counts, to `rtol` otherwise (an
empty bin's `NaN` equals `NaN`). A value-binned histogram may have `moved` pairs in a neighbouring
value bin, since the device and the host round a pair's value independently: each moves a count of at
most `mass` and a sum of at most `mass * value`. `parts` alternates sums and counts. Prints the largest
difference of each array that disagrees."""
function agrees(got, ref; rtol = 1e-10, moved = 0, value = 0.0, mass = 1.0)
    g, r = parts(got), parts(ref)
    length(g) == length(r) || return false
    ok = true
    for (k, (a, b)) in enumerate(zip(g, r))
        size(a) == size(b) || (println("    array $k: size $(size(a)) against $(size(b))"); return false)
        same = if moved == 0
            eltype(a) <: Integer ? a == b : isapprox(a, b; rtol, atol = rtol, nans = true)
        else
            unit = isodd(k) ? mass * value : mass
            d = abs.(Float64.(a) .- Float64.(b))
            sum(x -> isnan(x) ? 0.0 : x, d) <= rtol * sum(x -> isnan(x) ? 0.0 : abs(x), Float64.(b)) + 2 * moved * unit
        end
        if !same
            d = maximum(abs.(Float64.(a) .- Float64.(b)); init = 0.0)
            @printf("    array %d of %d (%s): max |Δ| = %.3e, max |ref| = %.3e\n", k, length(g),
                    eltype(a), d, maximum(abs.(Float64.(b)); init = 0.0))
            ok = false
        end
    end
    return ok
end

"""A tiled row: its kernel compiled at `predicted` bytes (at most `slack` more than `ptxas` reports),
within the budget, and the call agrees with serial."""
function tiled_row(name, pattern, predicted, run, ref_run; slack = 0, kw...)
    got, smem = compiled_smem(run)
    measured = entry_bytes(smem, pattern)
    check("$name: $pattern compiled", measured !== nothing)
    if measured !== nothing
        check("$name: ptxas $measured B, predicted $predicted B",
              measured <= predicted <= measured + slack && predicted <= BUDGET)
    end
    check("$name: agrees with serial", agrees(got, ref_run(); kw...))
    return nothing
end

"""A row one past the fit: the tiled kernel `pattern` is not compiled and the call agrees with serial."""
function past_row(name, pattern, run, ref_run; kw...)
    got, smem = compiled_smem(run)
    check("$name: $pattern not compiled", entry_bytes(smem, pattern) === nothing)
    check("$name: agrees with serial", agrees(got, ref_run(); kw...))
    return nothing
end

"""A row past every staged fit: the wide kernel `pattern` is compiled with no shared memory, no tiled
kernel is, and the call agrees with serial."""
function wide_row(name, pattern, run, ref_run; kw...)
    got, smem = compiled_smem(run)
    check("$name: $pattern compiled, 0 B", entry_bytes(smem, pattern) == 0)
    check("$name: no tiled kernel compiled", !any(k -> occursin("tiled", k), keys(smem)))
    check("$name: agrees with serial", agrees(got, ref_run(); kw...))
    return nothing
end

pts(::Type{FT}, D) where {FT} = rand(FT, D, N)
weights(::Type{FT}) where {FT} = FT(0.5) .+ rand(FT, N)
edges(::Type{FT}, hi, n) where {FT} = collect(range(FT(0), FT(hi); length = n + 1))
geom(D) = SF.HelperFunctions.FlatGeometry{D}()
# A value-binned row over value bins in [-1, 1] with pair weights below 1.5 each.
const VALUE_BINNED = (; moved = 2, value = 1.0, mass = 2.25)
widest_width(fits) = maximum(D for D in 2:64 if fits(D))

"""`_sf_launch_1d_batch_portable!` on `x` `(D, N)` (shared positions) or `(D, N, B)` and `u`
`(D, N, B)`, weighted: `(NMOM, NB, B)` device sums and counts of `F64`."""
function portable_1d(x, u, w, bins, NMOM)
    D, Np, Bu = size(u, 1), size(u, 2), size(u, 3)
    NB = length(bins) - 1
    out, cnt = CUDA.zeros(F64, NMOM, NB, Bu), CUDA.zeros(F64, NMOM, NB, Bu)
    GE._sf_launch_1d_batch_portable!(BE, out, cnt, CuArray(x), CuArray(u), NMOM == 1 ? OP : nothing,
        GE._gpu_digitizer(BE, bins, Val(NMOM == 1 ? :sf1d : :single_pass)), Np, NB, Bu, Val(NMOM),
        ndims(x) == 2, geom(D); weights = CuArray(w))
    CUDA.synchronize()
    return out, cnt
end

"""`_sf_launch_2d_batch_portable!` as [`portable_1d`](@ref), for `(NMOM, n_dist, n_val, B)` histograms."""
function portable_2d_batch(x, u, w, bins, vb, NMOM)
    D, Np, Bu = size(u, 1), size(u, 2), size(u, 3)
    nd, nv = length(bins) - 1, length(vb) - 1
    out, cnt = CUDA.zeros(F64, NMOM, nd, nv, Bu), CUDA.zeros(F64, NMOM, nd, nv, Bu)
    GE._sf_launch_2d_batch_portable!(BE, out, cnt, CuArray(x), CuArray(u), NMOM == 1 ? OP : nothing,
        GE._gpu_digitizer(BE, bins, Val(NMOM == 1 ? :joint2d : :single_pass_2d)),
        GE._value_digitizer(nothing, BE, vb), Np, nd, nv, Bu, Val(NMOM), ndims(x) == 2, geom(D),
        SFC.InvariantValueAxis(); weights = CuArray(w))
    CUDA.synchronize()
    return out, cnt
end

"""`_launch_joint_2d_portable!` on one weighted point list: `(n_dist, n_val)` sums and counts of `FT`."""
function portable_joint(x::AbstractMatrix{FT}, u, w, bins, vb) where {FT}
    nd, nv = length(bins) - 1, length(vb) - 1
    out, cnt = CUDA.zeros(FT, nd, nv), CUDA.zeros(FT, nd, nv)
    GE._launch_joint_2d_portable!(BE, 64, out, cnt, CuArray(x), CuArray(u), OP,
        GE._gpu_digitizer(BE, bins, Val(:joint2d)), GE._gpu_digitizer(BE, vb, Val(:value)),
        size(x, 2), nd + 1, nv + 1, geom(size(x, 1)); weights = CuArray(w))
    CUDA.synchronize()
    return Raw(out, cnt)
end

"""The serial weighted six-invariant 1-D batch histograms `(6, NB, B)`."""
function serial_sp_batch(x, u, bins, w)
    s = zeros(F64, SFC.SINGLE_PASS_N, length(bins) - 1, size(u, 3)); c = zeros(F64, size(s))
    SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = SER, weights = w)
    return Raw(s, c)
end

println("device=", CUDA.name(CUDA.device()), "  static budget=", BUDGET, " B  warp=", CAPS.warp)
Random.seed!(20260925)

# --- 1-D, per-slice positions: sf_tiled_1d_varying! at the widest width it fits, then past it ---
let bytes_v = (D, R) -> GE._sf_1d_varying_smem_bytes(F64, F64, F64, F64, D, D, 1, R)
    Dstar = widest_width(D -> SFC.gpu_static_smem_fits(CAPS, bytes_v(D, 1)))
    for D in (Dstar, Dstar + 1)
        x, u, w, bins = pts(F64, D), pts(F64, D), weights(F64), edges(F64, sqrt(D), 16)
        run = () -> Raw(map(vec, portable_1d(reshape(x, D, N, 1), reshape(u, D, N, 1), w, bins, 1))...)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, F64, RAW; backend = SER, weights = w)
        if D == Dstar
            R = GE._sf_fitting_width(r -> bytes_v(D, r), CAPS, 2)
            tiled_row("1d varying D=$D R=$R", "sf_tiled_1d_varying", bytes_v(D, R), run, ref)
        else
            wide_row("1d varying D=$D", "sf_wide_1d", run, ref)
        end
    end
end

# --- 1-D, shared positions: sf_tiled_1d_fixed! at each strip width the ladder reaches, then past it ---
let bytes_f = (D, W) -> GE._sf_1d_fixed_smem_bytes(F64, F64, F64, F64, D, D, 1, W)
    D4 = widest_width(D -> SFC.gpu_static_smem_fits(CAPS, bytes_f(D, 4)))
    D1 = widest_width(D -> SFC.gpu_static_smem_fits(CAPS, bytes_f(D, 1)))
    for D in unique((D4, D4 + 1, D1, D1 + 1))
        x, u, w, bins = pts(F64, D), rand(F64, D, N, B), weights(F64), edges(F64, sqrt(D), 16)
        run = () -> Raw(map(a -> reshape(a, 16, B), portable_1d(x, u, w, bins, 1))...)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, F64, RAW; backend = SER, weights = w)
        W = GE._sf_fitting_width(v -> bytes_f(D, v), CAPS, 4)
        if W > 0
            tiled_row("1d fixed batch D=$D W=$W", "sf_tiled_1d_fixed", bytes_f(D, W), run, ref)
        else
            wide_row("1d fixed batch D=$D", "sf_wide_1d", run, ref)
        end
    end
    x, u, w, bins = pts(F64, 3), rand(F64, 3, N, B), weights(F64), edges(F64, 2, 16)
    tiled_row("single pass fixed batch D=3", "sf_tiled_1d_fixed",
        GE._sf_1d_fixed_smem_bytes(F64, F64, F64, F64, 3, 3, 6, 1),
        () -> Raw(portable_1d(x, u, w, bins, 6)...), () -> serial_sp_batch(x, u, bins, w))
end

# --- single-pass 1-D tiled kernel ---
for D in (2, 3)
    x, u, w, bins = pts(F64, D), pts(F64, D), weights(F64), edges(F64, sqrt(D), 32)
    run = () -> begin
        out, cnt = CUDA.zeros(F64, SFC.SINGLE_PASS_N, 32), CUDA.zeros(F64, SFC.SINGLE_PASS_N, 32)
        GE._launch_single_pass_portable!(BE, 64, out, cnt, CuArray(x), CuArray(u),
            GE._gpu_digitizer(BE, bins, Val(:single_pass)), N, 33, geom(D); weights = CuArray(w))
        CUDA.synchronize()
        Raw(out, cnt)
    end
    ref = () -> Raw(values(SFC._dispatch_single_pass(SER, SFC.PointField{D}(), x, u, bins, F64; weights = w))...)
    tiled_row("single pass point D=$D", "_sf6_single_pass_kernel_tiled128", GE._sp1d_tiled_smem_bytes(F64, F64, D, D),
              run, ref)
end

# --- joint 2-D point: the widest histogram joint2d_smem_max admits, one value bin past it ---
let nd = 10, widest = SFC.joint2d_smem_max(BE, 2, 2, F64, F64, F64)
    nv = widest ÷ nd
    x, u, w, bins = pts(F64, 2), pts(F64, 2), weights(F64), edges(F64, 1.5, nd)
    for n in (nv, nv + 1)
        vb = collect(range(F64(-1), F64(1); length = n + 1))
        run = () -> portable_joint(x, u, w, bins, vb)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, F64; backend = SER, weights = w)
        if n == nv
            tiled_row("joint point $(nd)x$n (widest $widest)", "_sf2d_kernel_tiled128",
                      GE._joint2d_tiled_smem_bytes(F64, F64, F64, 2, 2, nd * n), run, ref; VALUE_BINNED...)
        else
            past_row("joint point $(nd)x$n", "_sf2d_kernel_tiled128", run, ref; VALUE_BINNED...)
        end
    end
    x32, u32, w32 = rand(Float32, 2, N), rand(Float32, 2, N), weights(Float32)
    b100, v100 = edges(Float32, 1.5, 100), collect(range(-1.0f0, 1.0f0; length = 101))
    past_row("joint point 100x100 Float32", "_sf2d_kernel_tiled128",
        () -> portable_joint(x32, u32, w32, b100, v100),
        () -> SFC.calculate_structure_function(OP, x32, u32, b100, v100, Float32; backend = SER, weights = w32);
        rtol = 1e-4, VALUE_BINNED...)
end

# --- joint 2-D batch: the shared histogram at its widest, then the staged global-atomic kernels ---
let nd = 10, bytes_s = nc -> GE._sf_2d_shared_smem_bytes(F64, F64, F64, F64, 2, 2, 1, nc)
    nv = GE._smem_max_cells(bytes_s, BUDGET, 2 * sizeof(F64)) ÷ nd
    xf, xv, u, w, bins = pts(F64, 2), rand(F64, 2, N, B), rand(F64, 2, N, B), weights(F64), edges(F64, 1.5, nd)
    for n in (nv, nv + 1), (label, x) in (("fixed", xf), ("varying", xv))
        vb = collect(range(F64(-1), F64(1); length = n + 1))
        run = () -> Raw(map(a -> reshape(a, nd, n, B), portable_2d_batch(x, u, w, bins, vb, 1))...)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, F64; backend = SER, weights = w)
        if n == nv
            tiled_row("joint $label batch $(nd)x$n", "sf_tiled_2d_shared", bytes_s(nd * n), run, ref; VALUE_BINNED...)
        elseif label == "fixed"
            W = GE._sf_tiled_2d_fixed_strip(CAPS, F64, F64, 2, 2)
            tiled_row("joint fixed batch $(nd)x$n W=$W", "sf_tiled_2d_fixed",
                      GE._sf_2d_fixed_smem_bytes(F64, F64, 2, 2, W), run, ref; VALUE_BINNED...)
        else
            tiled_row("joint varying batch $(nd)x$n", "sf_tiled_2d_varying",
                      GE._sf_2d_varying_smem_bytes(F64, F64, 2, 2), run, ref; VALUE_BINNED...)
        end
    end
    Dref = minimum(D for D in 2:64 if !SFC.gpu_static_smem_fits(CAPS, GE._sf_2d_varying_smem_bytes(F64, F64, D, D)))
    xr, xrf, ur = rand(F64, Dref, N, B), pts(F64, Dref), rand(F64, Dref, N, B)
    br, vr = edges(F64, sqrt(Dref), 8), collect(range(F64(-1), F64(1); length = 5))
    check("joint varying batch D=$Dref: no native plan",
          CE._cuda_2d_plan(CAPS, F64, F64, F64, F64, CuArray(w), geom(Dref), 1, 8, 4) === nothing)
    for (label, x) in (("varying", xr), ("fixed", xrf))
        ref = () -> SFC.calculate_structure_function(OP, x, ur, br, vr, F64; backend = SER, weights = w)
        wide_row("joint $label batch D=$Dref", "sf_wide_2d",
                 () -> Raw(map(a -> reshape(a, 8, 4, B), portable_2d_batch(x, ur, w, br, vr, 1))...), ref;
                 VALUE_BINNED...)
        check("joint $label batch D=$Dref: public entry agrees with serial",
              agrees(SFC.calculate_structure_function(OP, x, ur, br, vr, F64; backend = DEV, weights = w), ref();
                     VALUE_BINNED...))
    end
end

# --- single-pass 2-D, weighted: every accumulation mode, and the shared → typeplane boundary ---
function sp2d_row(nd, nv, D, w)
    cfg = GE._sp2d_accumulation_strategy(CAPS, nd, nv, D, D, F64, F64, F64)
    hc = GE._sp2d_sharedhist_compile_cells(cfg)
    pattern, predicted = if cfg.accum_mode === :shared
        "_sf6_sp2d_sharedhist", GE._sp2d_sharedhist_smem_bytes(F64, F64, F64, D, D, hc)
    elseif cfg.accum_mode === :typeplane
        "_sf6_sp2d_typeplane", GE._sp2d_typeplane_smem_bytes(F64, F64, F64, D, D, hc)
    else
        "_sf6_sp2d_directpartition", GE._sp2d_direct_smem_bytes(F64, D, D)
    end
    x, u, bins = pts(F64, D), pts(F64, D), edges(F64, sqrt(D), nd)
    vb = collect(range(F64(-1), F64(1); length = nv + 1))
    run = () -> begin
        out, cnt = CUDA.zeros(F64, SFC.SINGLE_PASS_N, nd, nv), CUDA.zeros(F64, SFC.SINGLE_PASS_N, nd, nv)
        GE._launch_single_pass_2d_portable!(BE, 64, out, cnt, CuArray(x), CuArray(u),
            GE._gpu_digitizer(BE, bins, Val(:single_pass_2d)), GE._value_digitizer(nothing, BE, vb),
            N, nd + 1, nv + 1, geom(D); weights = CuArray(w))
        CUDA.synchronize()
        Raw(out, cnt)
    end
    ref = () -> begin
        s = zeros(F64, SFC.SINGLE_PASS_N, nd, nv); c = zeros(F64, size(s))
        SFC.calculate_structure_functions_single_pass_2d!(s, c, x, u, bins, vb; backend = SER, weights = w)
        Raw(s, c)
    end
    tiled_row("sp2d $(nd)x$nv D=$D $(cfg.accum_mode) HC=$hc", pattern, predicted, run, ref;
              slack = SFC.GPU_SMEM_ALIGN, VALUE_BINNED...)
    return nothing
end

let w = weights(F64)
    mode_of = nv -> GE._sp2d_accumulation_strategy(CAPS, 10, nv, 2, 2, F64, F64, F64).accum_mode
    nv_shared = maximum(nv for nv in 2:200 if mode_of(nv) === :shared)
    check("sp2d 10x$(nv_shared) is the widest shared histogram", mode_of(nv_shared + 1) !== :shared)
    for (nd, nv, D) in ((10, nv_shared, 2), (10, nv_shared + 1, 2), (10, 8, 2), (30, 30, 2), (30, 30, 3), (60, 60, 2))
        sp2d_row(nd, nv, D, w)
    end
end

# --- 1-D shared-position batch, unweighted and two-wide: the field-strip kernel and its merges ---
for (k, FTb) in enumerate((Float32, Float64))
    SW = GE._batch_usmem_strip_w(CAPS, FTb)
    x, u, bins = rand(FTb, 2, N), rand(FTb, 2, N, 4SW + 1), edges(FTb, 1.5, 32)
    run = () -> SFC.calculate_structure_function(OP, x, u, bins, UInt32, RAW; backend = DEV)
    ref = () -> SFC.calculate_structure_function(OP, x, u, bins, UInt32, RAW; backend = SER)
    got, smem = compiled_smem(run)
    rows = [("_batch_fixed_x_usmem_priv", GE._batch_fixed_x_smem_bytes(FTb, SW)),
            ("_batch_merge_usmem_sums_grouped", SFC.gpu_localmem_bytes(FTb, GE.SF_GPU_TILED_WS))]
    k == 1 && push!(rows, ("_batch_merge_usmem_cnts_grouped", SFC.gpu_localmem_bytes(UInt32, GE.SF_GPU_TILED_WS)))
    for (pattern, predicted) in rows
        check("fixed batch $FTb SW=$SW: $pattern at $predicted B", entry_bytes(smem, pattern) == predicted)
    end
    check("fixed batch $FTb SW=$SW: agrees with serial", agrees(got, ref(); rtol = FTb == Float32 ? 1e-5 : 1e-10))
end

# --- native CUDA 1-D: the public point entry at the plan the native plan function returns ---
plan_params(::CE.CUDA1DPlan{W, F, M, T, R, S, H, C}) where {W, F, M, T, R, S, H, C} = (; TILE = T, S, H, CST = C)
plan_params(::CE.CUDA2DPlan{W, F, M, T, C}) where {W, F, M, T, C} = (; TILE = T, CST = C)

for (D, weighted) in ((2, false), (3, false), (3, true), (5, false), (6, false))
    x, u, bins = pts(F64, D), pts(F64, D), edges(F64, sqrt(D), 64)
    w = weighted ? weights(F64) : nothing
    CT = weighted ? F64 : UInt32
    kw = weighted ? (; weights = w) : (;)
    run = () -> SFC.calculate_structure_function(OP, x, u, bins, CT, RAW; backend = DEV, kw...)
    ref = () -> SFC.calculate_structure_function(OP, x, u, bins, CT, RAW; backend = SER, kw...)
    plan = CE._cuda_1d_plan(CAPS, F64, F64, F64, CT, weighted ? CuArray(w) : SFC.NoWeights(), geom(D), 64, 1)
    tag = weighted ? " weighted" : ""
    if plan === nothing
        past_row("native 1d D=$D$tag, no plan", "_cuda_sf_1d_kernel", run, ref)
    else
        p = plan_params(plan)
        tiled_row("native 1d D=$D$tag TILE=$(p.TILE) S=$(p.S)", "_cuda_sf_1d_kernel",
            CE._cuda_1d_smem_bytes(F64, F64, F64, p.CST, D, D, 1, p.TILE, p.S, p.H), run, ref)
    end
end

# --- native CUDA 2-D: its static staging at the tile its plan picks, and the widths it has no plan for ---
for (FT, D, weighted) in ((Float32, 2, false), (F64, 3, true), (F64, 6, false), (F64, 7, false))
    x, u, bins = rand(FT, D, N), rand(FT, D, N), edges(FT, sqrt(D), 16)
    vb = collect(range(FT(-1), FT(1); length = 11))
    w = weighted ? weights(FT) : nothing
    CT = weighted ? FT : UInt32
    kw = weighted ? (; weights = w) : (;)
    run = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, CT; backend = DEV, kw...)
    ref = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, CT; backend = SER, kw...)
    plan = CE._cuda_2d_plan(CAPS, FT, FT, FT, CT, weighted ? CuArray(w) : SFC.NoWeights(), geom(D), 1, 16, 10)
    tag = "$FT D=$D$(weighted ? " weighted" : "")"
    tol = FT == Float32 ? (; rtol = 1e-5, moved = 2, value = 1.0) : VALUE_BINNED
    if plan === nothing
        past_row("native joint $tag, no plan", "_cuda_sf_2d_kernel", run, ref; tol...)
    else
        p = plan_params(plan)
        tiled_row("native joint $tag TILE=$(p.TILE)", "_cuda_sf_2d_kernel",
            4 * SFC.gpu_localmem_bytes(FT, D * p.TILE), run, ref; tol...)
    end
end

isempty(failures) || error("shared-memory budget: $(length(failures)) rows failed:\n  " * join(failures, "\n  "))
println("SMEM_BUDGET_OK")
