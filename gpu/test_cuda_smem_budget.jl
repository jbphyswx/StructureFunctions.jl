using CUDA: CUDA
using Random: Random
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

"""Whether each array of `got` equals `ref`'s, integers exactly, up to `moved` pairs changing value bin."""
function agrees(got, ref; rtol = 1e-10, moved = 0, value = 0.0, mass = 1.0)
    g, r = parts(got), parts(ref)
    length(g) == length(r) || return false
    for (k, (a, b)) in enumerate(zip(g, r))
        size(a) == size(b) || return false
        same = if moved == 0
            eltype(a) <: Integer ? a == b : isapprox(a, b; rtol, atol = rtol, nans = true)
        else
            unit = isodd(k) ? mass * value : mass
            d = abs.(Float64.(a) .- Float64.(b))
            sum(x -> isnan(x) ? 0.0 : x, d) <= rtol * sum(x -> isnan(x) ? 0.0 : abs(x), Float64.(b)) + 2 * moved * unit
        end
        same || return false
    end
    return true
end

"""A tiled case: `pattern` compiled at `predicted` bytes, at most `slack` over `ptxas`, within budget, as serial."""
function tiled_row(name, pattern, predicted, run, ref_run; slack = 0, kw...)
    Test.@testset "$name" begin
        got, smem = compiled_smem(run)
        measured = entry_bytes(smem, pattern)
        Test.@test measured !== nothing
        measured === nothing || Test.@test measured <= predicted <= measured + slack
        Test.@test predicted <= BUDGET
        Test.@test agrees(got, ref_run(); kw...)
    end
end

"""A case one past the fit: the tiled kernel `pattern` is not compiled and the call agrees with serial."""
function past_row(name, pattern, run, ref_run; kw...)
    Test.@testset "$name" begin
        got, smem = compiled_smem(run)
        Test.@test entry_bytes(smem, pattern) === nothing
        Test.@test agrees(got, ref_run(); kw...)
    end
end

"""A case past every staged fit: the wide kernel `pattern` at 0 bytes, no tiled kernel, agreeing with serial."""
function wide_row(name, pattern, run, ref_run; kw...)
    Test.@testset "$name" begin
        got, smem = compiled_smem(run)
        Test.@test entry_bytes(smem, pattern) == 0
        Test.@test !any(k -> occursin("tiled", k), keys(smem))
        Test.@test agrees(got, ref_run(); kw...)
    end
end

pts(::Type{FT}, D) where {FT} = rand(FT, D, N)
weights(::Type{FT}) where {FT} = FT(0.5) .+ rand(FT, N)
edges(::Type{FT}, hi, n) where {FT} = collect(range(FT(0), FT(hi); length = n + 1))
geom(D) = SF.HelperFunctions.FlatGeometry{D}()
# A value-binned case over value bins in [-1, 1] with pair weights below 1.5 each.
const VALUE_BINNED = (; moved = 2, value = 1.0, mass = 2.25)
widest_width(fits) = maximum(D for D in 2:64 if fits(D))

"""`(NMOM, NB, B)` device sums and counts of `_sf_launch_1d_batch_portable!`, weighted, `x` shared when a matrix."""
function portable_1d(x, u, w, bins, NMOM)
    D, Np, Bu = size(u, 1), size(u, 2), size(u, 3)
    NB = length(bins) - 1
    out, cnt = CUDA.zeros(F64, NMOM, NB, Bu), CUDA.zeros(F64, NMOM, NB, Bu)
    GE._sf_launch_1d_batch_portable!(BE, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u),
        NMOM == 1 ? OP : SFT.SinglePassInvariants(),
        GE._gpu_digitizer(BE, bins, Val(NMOM == 1 ? :sf1d : :single_pass)), Np, NB, Bu,
        ndims(x) == 2, geom(D); weights = CUDA.CuArray(w))
    CUDA.synchronize()
    return out, cnt
end

"""`_sf_launch_2d_batch_portable!` as [`portable_1d`](@ref), for `(NMOM, n_dist, n_val, B)` histograms."""
function portable_2d_batch(x, u, w, bins, vb, NMOM)
    D, Np, Bu = size(u, 1), size(u, 2), size(u, 3)
    nd, nv = length(bins) - 1, length(vb) - 1
    out, cnt = CUDA.zeros(F64, NMOM, nd, nv, Bu), CUDA.zeros(F64, NMOM, nd, nv, Bu)
    GE._sf_launch_2d_batch_portable!(BE, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u),
        NMOM == 1 ? OP : SFT.SinglePassInvariants(),
        GE._gpu_digitizer(BE, bins, Val(NMOM == 1 ? :joint2d : :single_pass_2d)),
        GE._value_digitizer(nothing, BE, vb), Np, nd, nv, Bu, ndims(x) == 2, geom(D),
        SFC.InvariantValueAxis(); weights = CUDA.CuArray(w))
    CUDA.synchronize()
    return out, cnt
end

"""`_launch_joint_2d_portable!` on one weighted point list: `(n_dist, n_val)` sums and counts of `FT`."""
function portable_joint(x::AbstractMatrix{FT}, u, w, bins, vb) where {FT}
    nd, nv = length(bins) - 1, length(vb) - 1
    out, cnt = CUDA.zeros(FT, nd, nv), CUDA.zeros(FT, nd, nv)
    GE._launch_joint_2d_portable!(BE, 64, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u), OP,
        GE._gpu_digitizer(BE, bins, Val(:joint2d)), GE._gpu_digitizer(BE, vb, Val(:value)),
        size(x, 2), nd + 1, nv + 1, geom(size(x, 1)); weights = CUDA.CuArray(w))
    CUDA.synchronize()
    return Raw(out, cnt)
end

"""The serial weighted six-invariant 1-D batch histograms `(6, NB, B)`."""
function serial_sp_batch(x, u, bins, w)
    s = zeros(F64, SFC.SINGLE_PASS_N, length(bins) - 1, size(u, 3)); c = zeros(F64, size(s))
    SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = SER, weights = w)
    return Raw(s, c)
end

"""A weighted portable single-pass 2-D case on its accumulation strategy's kernel, or the global-atomic one."""
function sp2d_row(nd, nv, D, w)
    cfg = GE._sp2d_accumulation_strategy(CAPS, nd, nv, D, D, F64, F64, F64)
    x, u, bins = pts(F64, D), pts(F64, D), edges(F64, sqrt(D), nd)
    vb = collect(range(F64(-1), F64(1); length = nv + 1))
    run = () -> begin
        out, cnt = CUDA.zeros(F64, SFC.SINGLE_PASS_N, nd, nv), CUDA.zeros(F64, SFC.SINGLE_PASS_N, nd, nv)
        GE._launch_single_pass_2d_portable!(BE, 64, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u),
            GE._gpu_digitizer(BE, bins, Val(:single_pass_2d)), GE._value_digitizer(nothing, BE, vb),
            N, nd + 1, nv + 1, geom(D); weights = CUDA.CuArray(w))
        CUDA.synchronize()
        Raw(out, cnt)
    end
    ref = () -> begin
        s = zeros(F64, SFC.SINGLE_PASS_N, nd, nv); c = zeros(F64, size(s))
        SFC.calculate_structure_functions_single_pass_2d!(s, c, x, u, bins, vb; backend = SER, weights = w)
        Raw(s, c)
    end
    if cfg === nothing
        wide_row("sp2d $(nd)x$nv D=$D global atomics", "_sf_single_pass_2d_kernel", run, ref; VALUE_BINNED...)
        return nothing
    end
    hc = GE._sp2d_sharedhist_compile_cells(cfg)
    pattern, predicted = cfg.accum_mode === :shared ?
        ("_sf6_sp2d_sharedhist", GE._sp2d_sharedhist_smem_bytes(F64, F64, F64, D, D, hc)) :
        ("_sf6_sp2d_typeplane", GE._sp2d_typeplane_smem_bytes(F64, F64, F64, D, D, hc))
    tiled_row("sp2d $(nd)x$nv D=$D $(cfg.accum_mode) HC=$hc", pattern, predicted, run, ref;
              slack = SFC.GPU_SMEM_ALIGN, VALUE_BINNED...)
    return nothing
end

plan_params(::CE.CUDA1DPlan{W, F, M, T, R, S, H, C, Q}) where {W, F, M, T, R, S, H, C, Q} =
    (; TILE = T, S, H, CST = C, Q, kernel = Q == 0 ? "_cuda_sf_1d_kernel" : "_cuda_sf_1d_queued_kernel")
plan_params(::CE.CUDA2DPlan{W, F, M, T, C, NP}) where {W, F, M, T, C, NP} =
    (; TILE = T, CST = C, NP, kernel = "_cuda_sf_2d_kernel")
plan_params(::CE.CUDA2DGlobalPlan{W, F, M, T}) where {W, F, M, T} =
    (; TILE = T, NP = 0, kernel = "_cuda_sf_2d_global_kernel")

# (fit side of the joint batch, positions)
const JOINT_BATCH_CASES = ((:fit, "fixed"), (:past, "fixed"), (:past, "varying"))
# (distance bins, value bins, width): the widest shared histogram, one past it, and one no on-chip mode holds
const SP2D_CASES = ((10, :widest_shared, 2), (10, :past_shared, 2), (60, 60, 2))
# (width, weighted)
const NATIVE_1D_CASES = ((2, false), (3, true))
# (element type, width)
const NATIVE_SP1D_CASES = ((Float32, 2), (F64, 3))
# (element type, width, weighted)
const NATIVE_2D_CASES = ((Float32, 2, false), (F64, 3, true))

Test.@testset "static shared memory against ptxas" begin
    Random.seed!(20260925)

    # The portable 1-D kernel for per-slice positions at the widest width it fits, then past it.
    Test.@testset "1-D varying" begin
        bytes_v = (D, R) -> GE._sf_1d_varying_smem_bytes(F64, F64, F64, F64, D, D, 1, R)
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

    # The portable 1-D kernel for shared positions at each strip width the ladder reaches, then past it.
    Test.@testset "1-D fixed" begin
        bytes_f = (D, W) -> GE._sf_1d_fixed_smem_bytes(F64, F64, F64, F64, D, D, 1, W)
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

    # The portable single-pass 1-D tiled kernel.
    Test.@testset "single-pass 1-D" begin
        D = 3
        x, u, w, bins = pts(F64, D), pts(F64, D), weights(F64), edges(F64, sqrt(D), 32)
        run = () -> begin
            out, cnt = CUDA.zeros(F64, SFC.SINGLE_PASS_N, 32), CUDA.zeros(F64, SFC.SINGLE_PASS_N, 32)
            GE._launch_single_pass_portable!(BE, 64, out, cnt, CUDA.CuArray(x), CUDA.CuArray(u),
                GE._gpu_digitizer(BE, bins, Val(:single_pass)), N, 33, geom(D); weights = CUDA.CuArray(w))
            CUDA.synchronize()
            Raw(out, cnt)
        end
        ref = () -> Raw(values(SFC._dispatch_single_pass(SER, SFC.PointField{D}(), x, u, bins, F64; weights = w,
                                                         geometry = geom(D)))...)
        tiled_row("single pass point D=$D", "_sf6_single_pass_kernel_tiled128",
                  GE._sp1d_tiled_smem_bytes(F64, F64, D, D), run, ref)
    end

    # The portable joint 2-D point kernel at the widest histogram `joint2d_smem_max` admits, then one value bin past it.
    Test.@testset "joint 2-D point" begin
        nd, widest = 10, SFC.joint2d_smem_max(BE, 2, 2, F64, F64, F64)
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
    end

    # The portable joint 2-D batch kernels: shared histogram at its widest, staged global atomics past it, then wide.
    Test.@testset "joint 2-D batch" begin
        nd = 10
        bytes_s = nc -> GE._sf_2d_shared_smem_bytes(F64, F64, F64, F64, 2, 2, 1, nc)
        nv = GE._smem_max_cells(bytes_s, BUDGET, 2 * sizeof(F64)) ÷ nd
        xf, xv, u, w, bins = pts(F64, 2), rand(F64, 2, N, B), rand(F64, 2, N, B), weights(F64), edges(F64, 1.5, nd)
        for (side, label) in JOINT_BATCH_CASES
            n = side === :fit ? nv : nv + 1
            x = label == "fixed" ? xf : xv
            vb = collect(range(F64(-1), F64(1); length = n + 1))
            run = () -> Raw(map(a -> reshape(a, nd, n, B), portable_2d_batch(x, u, w, bins, vb, 1))...)
            ref = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, F64; backend = SER, weights = w)
            if side === :fit
                tiled_row("joint $label batch $(nd)x$n", "sf_tiled_2d_shared", bytes_s(nd * n), run, ref;
                          VALUE_BINNED...)
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
        xr, ur = rand(F64, Dref, N, B), rand(F64, Dref, N, B)
        br, vr = edges(F64, sqrt(Dref), 8), collect(range(F64(-1), F64(1); length = 5))
        Test.@test CE._cuda_2d_plan(CAPS, F64, F64, F64, F64, CUDA.CuArray(w), geom(Dref), OP, 8, 4) === nothing
        ref = () -> SFC.calculate_structure_function(OP, xr, ur, br, vr, F64; backend = SER, weights = w)
        wide_row("joint varying batch D=$Dref", "sf_wide_2d",
                 () -> Raw(map(a -> reshape(a, 8, 4, B), portable_2d_batch(xr, ur, w, br, vr, 1))...), ref;
                 VALUE_BINNED...)
        Test.@test agrees(SFC.calculate_structure_function(OP, xr, ur, br, vr, F64; backend = DEV, weights = w), ref();
                          VALUE_BINNED...)
    end

    # The weighted portable single-pass 2-D kernels across the shared, type-plane and global-atomic modes.
    Test.@testset "single-pass 2-D" begin
        w = weights(F64)
        mode_of = nv -> (cfg = GE._sp2d_accumulation_strategy(CAPS, 10, nv, 2, 2, F64, F64, F64);
                         cfg === nothing ? nothing : cfg.accum_mode)
        nv_shared = maximum(nv for nv in 2:200 if mode_of(nv) === :shared)
        Test.@test mode_of(nv_shared + 1) !== :shared
        for (nd, nv, D) in SP2D_CASES
            n = nv === :widest_shared ? nv_shared : nv === :past_shared ? nv_shared + 1 : nv
            sp2d_row(nd, n, D, w)
        end
    end

    # The fixed-position field-strip batch kernel and its merges, through their launcher.
    Test.@testset "1-D shared-position strip batch $FTb" for (k, FTb) in enumerate((Float32, Float64))
        SW = GE._batch_usmem_strip_w(CAPS, FTb)
        Bs = 4SW + 1
        x, u, bins = rand(FTb, 2, N), rand(FTb, 2, N, Bs), edges(FTb, 1.5, 32)
        run = () -> begin
            s, c = CUDA.zeros(FTb, 32, Bs), CUDA.zeros(UInt32, 32, Bs)
            xd, ud = GE._stage_batch_device(BE, x, u; fixed_x = true)
            GE._launch_batch_fixed_x_sf!(BE, s, c, xd, ud, OP, N, Bs, GE._gpu_digitizer(BE, bins, Val(:sf1d)), 32,
                                         geom(2))
            Raw(s, c)
        end
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, UInt32, RAW; backend = SER)
        got, smem = compiled_smem(run)
        rows = [("_batch_fixed_x_usmem_priv", GE._batch_fixed_x_smem_bytes(FTb, SW)),
                ("_batch_merge_usmem_sums_grouped", SFC.gpu_localmem_bytes(FTb, GE.SF_GPU_TILED_WS))]
        k == 1 && push!(rows, ("_batch_merge_usmem_cnts_grouped", SFC.gpu_localmem_bytes(UInt32, GE.SF_GPU_TILED_WS)))
        for (pattern, predicted) in rows
            Test.@test (pattern, entry_bytes(smem, pattern)) == (pattern, predicted)
        end
        Test.@test agrees(got, ref(); rtol = FTb == Float32 ? 1e-5 : 1e-10)
    end

    # The native 1-D kernel through the public point entry at the plan the native plan function returns.
    Test.@testset "native 1-D D=$D weighted=$weighted" for (D, weighted) in NATIVE_1D_CASES
        x, u, bins = pts(F64, D), pts(F64, D), edges(F64, sqrt(D), 64)
        w = weighted ? weights(F64) : nothing
        CT = weighted ? F64 : UInt32
        kw = weighted ? (; weights = w) : (;)
        run = () -> SFC.calculate_structure_function(OP, x, u, bins, CT, RAW; backend = DEV, kw...)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, CT, RAW; backend = SER, kw...)
        choice = CE._cuda_1d_plan(CAPS, F64, F64, F64, CT, weighted ? CUDA.CuArray(w) : SFC.NoWeights(), geom(D), 64,
                                  OP)
        if choice === nothing
            past_row("no plan", "_cuda_sf_1d_kernel", run, ref)
        else
            Test.@test N * (N - 1) ÷ 2 < choice.choose_from
            p = plan_params(first(choice.candidates(N, 1, false, N * (N - 1) ÷ 2, 1.0)))
            tiled_row("TILE=$(p.TILE) S=$(p.S) Q=$(p.Q)", p.kernel,
                CE._cuda_1d_smem_bytes(F64, F64, F64, p.CST, D, D, 1, p.TILE, p.S, p.H, p.Q), run, ref)
        end
    end

    # The native 1-D kernel of six moments: a call too small to estimate takes its group's unestimated plan.
    Test.@testset "native 1-D single pass $FT D=$D" for (FT, D) in NATIVE_SP1D_CASES
        x, u, bins = pts(FT, D), pts(FT, D), edges(FT, sqrt(D), 64)
        run = () -> SFC.calculate_structure_functions_single_pass(x, u, bins, UInt32, RAW; backend = DEV)
        ref = () -> SFC.calculate_structure_functions_single_pass(x, u, bins, UInt32, RAW; backend = SER)
        choice = CE._cuda_1d_plan(CAPS, FT, FT, FT, UInt32, SFC.NoWeights(), geom(D), 64, SFT.SinglePassInvariants())
        Test.@test N * (N - 1) ÷ 2 < choice.choose_from
        p = plan_params(first(choice.candidates(N, 1, false, N * (N - 1) ÷ 2, 1.0)))
        tiled_row("TILE=$(p.TILE) S=$(p.S) Q=$(p.Q)", p.kernel,
            CE._cuda_1d_smem_bytes(FT, FT, FT, p.CST, D, D, 6, p.TILE, p.S, p.H, p.Q), run, ref;
            rtol = FT == Float32 ? 1e-5 : 1e-10)
    end

    # The native 2-D kernel: a call too small to estimate takes its group's unestimated plan, staged at its tile.
    Test.@testset "native joint $FT D=$D weighted=$weighted" for (FT, D, weighted) in NATIVE_2D_CASES
        x, u, bins = rand(FT, D, N), rand(FT, D, N), edges(FT, sqrt(D), 16)
        vb = collect(range(FT(-1), FT(1); length = 11))
        w = weighted ? weights(FT) : nothing
        CT = weighted ? FT : UInt32
        kw = weighted ? (; weights = w) : (;)
        run = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, CT; backend = DEV, kw...)
        ref = () -> SFC.calculate_structure_function(OP, x, u, bins, vb, CT; backend = SER, kw...)
        plan = CE._cuda_2d_plan(CAPS, FT, FT, FT, CT, weighted ? CUDA.CuArray(w) : SFC.NoWeights(), geom(D), OP, 16, 10)
        tol = FT == Float32 ? (; rtol = 1e-5, moved = 2, value = 1.0) : VALUE_BINNED
        if plan === nothing
            past_row("no plan", "_cuda_sf_2d_kernel", run, ref; tol...)
        else
            Test.@test N * (N - 1) ÷ 2 < plan.choose_from
            p = plan_params(first(plan.candidates(N, 1, false, N * (N - 1) ÷ 2, 1.0)))
            tiled_row("TILE=$(p.TILE) NP=$(p.NP)", p.kernel, 4 * SFC.gpu_localmem_bytes(FT, D * p.TILE), run, ref;
                      tol...)
        end
    end
end
