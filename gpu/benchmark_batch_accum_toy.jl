#!/usr/bin/env julia
"""
    benchmark_batch_accum_toy.jl

Batch **memory strategy** toy — both arms use **one host kernel launch**.

Question: given a single pair schedule (as in production snapshot SF), is inner batch
work cheaper **direct to VRAM** or via **block smem strip buffer + flush**?

**NOT tested here:** v1 `multi_pass_strip` host relaunch (replay pair grid per strip/pass).
That anti-pattern is opt-in only (`SHOW_V1=1`).

---

### `fused_vram` (VRAM inner-b target)

```text
one launch
  for each pair (grid-stride, once):
    for b in 1:B
      accumulate P typeplane passes in registers
      flush 2 global atomics → output[bin, b] in VRAM
```

`pair_traversals = pairs` (each pair visited once).

### `fused_block_smem` (production-shaped block smem)

Block smem `(NB, W)` merges contributions from **many pairs** before flushing to VRAM.
That requires revisiting the pair loop once per `(type pass, batch strip)` **inside the
same kernel** — not a host relaunch, but not a single literal pass over pairs either.

```text
one launch
  for pass in 1:P
    for strip in 1:ceil(B/W)
      zero block smem[NB × W]
      for each pair (grid-stride):
        accumulate W batch cols into smem
      flush smem → VRAM
```

`pair_traversals = pairs × P × ceil(B/W)`.

### `fused_vram_private` (block-private global partials + merge)

Same pair schedule as `fused_vram`, but atomics target `partial[bin, b, block_id]` in VRAM
(contention only within a thread block), then a merge kernel sums over blocks.

VRAM cost: `2 × NB × B × nblocks × sizeof(Float32)` total (printed at startup).
Per-block slab is `2 × NB × B × 4` bytes (~1.3 MiB at B=8064, ~13 MiB at B=80000); with
~1024 blocks that is ~1.3–13 GiB — feasible on 80 GiB GPUs. This is **not** smem-sized
`(NB, W)`; it privatizes the **full batch axis** per block to kill cross-block atomics.

Same total updates `(pair × b × pass)`. Checksums must match between arms.

---

    julia --project=gpu gpu/benchmark_batch_accum_toy.jl
    N=256 B=80000 P=4 julia --project=gpu gpu/benchmark_batch_accum_toy.jl

CPU always. CUDA when `SLURM_JOB_ID` or `ALLOW_CUDA_BENCH=1`.
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @kernel, @localmem, @synchronize
using Printf: @printf
using Random: Random

const TOY_NB = 20
const TOY_MAX_STRIP = 32
const TOY_SHMEM = TOY_NB * TOY_MAX_STRIP

@inline function _toy_bin(i::Int, j::Int, NB::Int)
    return (i * 31 + j * 17) % NB + 1
end

@inline function _toy_val(i::Int, j::Int, b::Int, p::Int)
    return sin(i * 0.001f0 + j * 0.002f0 + b * 0.01f0 + p * 0.1f0)
end

@inline function _toy_pair_from_linear(k::Int, N::Int)
    lo = 1
    hi = N - 1
    while lo < hi
        mid = (lo + hi) >>> 1
        row_start = (mid - 1) * N - (mid - 1) * mid ÷ 2
        row_end = row_start + (N - mid)
        if k > row_end
            lo = mid + 1
        else
            hi = mid
        end
    end
    i = lo
    j = i + (k - ((i - 1) * N - (i - 1) * i ÷ 2))
    return i, j
end

@inline function _toy_strip_b_base(W::Int, strip_id::Int)
    return (strip_id - 1) * W + 1
end

@inline function _toy_strip_bw(B::Int, W::Int, strip_id::Int)
    return min(W, B - _toy_strip_b_base(W, strip_id) + 1)
end

@inline function _toy_pass_from_chunk(chunk_id::Int, B::Int, W::Int)
    return (chunk_id - 1) ÷ cld(B, W) + 1
end

@inline function _toy_strip_from_chunk(chunk_id::Int, B::Int, W::Int)
    return (chunk_id - 1) % cld(B, W) + 1
end

# ---------------------------------------------------------------------------
# fused_vram — one pair traversal; inner b; direct VRAM output
# ---------------------------------------------------------------------------

@kernel function _toy_fused_vram!(
    output,
    @Const(u_table),
    N::Int,
    NB::Int,
    B::Int,
    P::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    total_pairs = N * (N - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _toy_pair_from_linear(k, N)
        bin = _toy_bin(i, j, NB)
        @inbounds for b in 1:B
            acc = 0.0f0
            cnt = UInt32(0)
            for pass_id in 1:P
                val = _toy_val(i, j, b, pass_id) + u_table[i, b] - u_table[j, b]
                acc += val
                cnt += UInt32(1)
            end
            @atomic output[bin, b] += acc
            @atomic output[NB + bin, b] += Float32(cnt)
        end
        k += nworkers
    end
end

function _launch_fused_vram!(backend, output, u_dev, N, B, P, nworkers, wg)
    NB = TOY_NB
    t = @elapsed begin
        kernel! = _toy_fused_vram!(backend, wg)
        kernel!(output, u_dev, N, NB, B, P, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end
    return t
end

# ---------------------------------------------------------------------------
# fused_vram_private — block-private global partials, then merge
# ---------------------------------------------------------------------------

@kernel function _toy_fused_vram_private!(
    partial,
    @Const(u_table),
    N::Int,
    NB::Int,
    B::Int,
    P::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    block_id = @index(Group, Linear)
    total_pairs = N * (N - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _toy_pair_from_linear(k, N)
        bin = _toy_bin(i, j, NB)
        @inbounds for b in 1:B
            acc = 0.0f0
            cnt = UInt32(0)
            for pass_id in 1:P
                val = _toy_val(i, j, b, pass_id) + u_table[i, b] - u_table[j, b]
                acc += val
                cnt += UInt32(1)
            end
            @atomic partial[bin, b, block_id] += acc
            @atomic partial[NB + bin, b, block_id] += Float32(cnt)
        end
        k += nworkers
    end
end

@kernel function _toy_merge_private!(
    output,
    @Const(partial),
    NB::Int,
    B::Int,
    nblocks::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    total = NB * B * nblocks
    t = worker
    while t <= total
        rem = (t - 1) % (NB * B)
        bin = rem % NB + 1
        b = rem ÷ NB + 1
        blk = (t - 1) ÷ (NB * B) + 1
        @inbounds begin
            @atomic output[bin, b] += partial[bin, b, blk]
            @atomic output[NB + bin, b] += partial[NB + bin, b, blk]
        end
        t += nworkers
    end
end

function _nblocks(nworkers::Int, wg::Int)
    return cld(nworkers, wg)
end

function _private_partial_bytes(B::Int, nblocks::Int)
    return 2 * TOY_NB * B * nblocks * sizeof(Float32)
end

function _private_slab_bytes(B::Int)
    return 2 * TOY_NB * B * sizeof(Float32)
end

function _make_partial!(backend, B, nblocks)
    partial = zeros(Float32, 2 * TOY_NB, B, nblocks)
    return KA.adapt(backend, partial)
end

function _launch_fused_vram_private!(backend, output, partial, u_dev, N, B, P, nworkers, wg)
    NB = TOY_NB
    nblocks = _nblocks(nworkers, wg)
    t = @elapsed begin
        fill!(partial, 0.0f0)
        KA.synchronize(backend)
        acc! = _toy_fused_vram_private!(backend, wg)
        acc!(partial, u_dev, N, NB, B, P, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
        merge! = _toy_merge_private!(backend, wg)
        merge!(output, partial, NB, B, nblocks, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end
    return t
end

# ---------------------------------------------------------------------------
# fused_block_smem — one launch; block smem strip; pair loop per (pass, strip)
# ---------------------------------------------------------------------------

@kernel function _toy_fused_block_smem!(
    output,
    @Const(u_table),
    N::Int,
    NB::Int,
    B::Int,
    P::Int,
    W::Int,
    nworkers::Int,
)
    shared_s = @localmem Float32 (TOY_SHMEM,)
    shared_c = @localmem UInt32 (TOY_SHMEM,)
    g = @index(Global, Linear)
    lid = @index(Local, Linear)

    for chunk_id in 1:(P * cld(B, W))
        if lid == 1
            @inbounds for s in 1:(NB * _toy_strip_bw(B, W, _toy_strip_from_chunk(chunk_id, B, W)))
                shared_s[s] = 0.0f0
                shared_c[s] = UInt32(0)
            end
        end
        @synchronize()

        k = g
        total_pairs = N * (N - 1) ÷ 2
        while k <= total_pairs
            i, j = _toy_pair_from_linear(k, N)
            bin = _toy_bin(i, j, NB)
            @inbounds for col in 1:_toy_strip_bw(B, W, _toy_strip_from_chunk(chunk_id, B, W))
                b = _toy_strip_b_base(W, _toy_strip_from_chunk(chunk_id, B, W)) + col - 1
                val = _toy_val(i, j, b, _toy_pass_from_chunk(chunk_id, B, W)) +
                      u_table[i, b] - u_table[j, b]
                idx = bin + (col - 1) * NB
                @atomic shared_s[idx] += val
                @atomic shared_c[idx] += UInt32(1)
            end
            k += nworkers
        end
        @synchronize()

        if lid == 1
            for slot in 1:(NB * _toy_strip_bw(B, W, _toy_strip_from_chunk(chunk_id, B, W)))
                col = (slot - 1) ÷ NB + 1
                bin = (slot - 1) % NB + 1
                b = _toy_strip_b_base(W, _toy_strip_from_chunk(chunk_id, B, W)) + col - 1
                idx = bin + (col - 1) * NB
                @atomic output[bin, b] += shared_s[idx]
                if shared_c[idx] != UInt32(0)
                    @atomic output[NB + bin, b] += Float32(shared_c[idx])
                end
            end
        end
        @synchronize()
    end
end

function _launch_fused_block_smem!(backend, output, u_dev, N, B, P, W, nworkers, wg)
    NB = TOY_NB
    t = @elapsed begin
        kernel! = _toy_fused_block_smem!(backend, wg)
        kernel!(output, u_dev, N, NB, B, P, W, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end
    return t
end

# ---------------------------------------------------------------------------
# v1 host relaunch (rejected) — opt-in reference only
# ---------------------------------------------------------------------------

@kernel function _toy_v1_host_relaunch!(
    output,
    @Const(u_table),
    N::Int,
    NB::Int,
    b_base::Int,
    bw::Int,
    pass_id::Int,
    nworkers::Int,
)
    shared_s = @localmem Float32 (TOY_SHMEM,)
    shared_c = @localmem UInt32 (TOY_SHMEM,)
    g = @index(Global, Linear)
    lid = @index(Local, Linear)

    if lid == 1
        @inbounds for s in 1:(NB * bw)
            shared_s[s] = 0.0f0
            shared_c[s] = UInt32(0)
        end
    end
    @synchronize()

    k = g
    total_pairs = N * (N - 1) ÷ 2
    while k <= total_pairs
        i, j = _toy_pair_from_linear(k, N)
        bin = _toy_bin(i, j, NB)
        @inbounds for col in 1:bw
            b = b_base + col - 1
            val = _toy_val(i, j, b, pass_id) + u_table[i, b] - u_table[j, b]
            idx = bin + (col - 1) * NB
            @atomic shared_s[idx] += val
            @atomic shared_c[idx] += UInt32(1)
        end
        k += nworkers
    end
    @synchronize()

    if lid == 1
        n_flush = NB * bw
        for slot in 1:n_flush
            col = (slot - 1) ÷ NB + 1
            bin = (slot - 1) % NB + 1
            b = b_base + col - 1
            idx = bin + (col - 1) * NB
            @atomic output[bin, b] += shared_s[idx]
            if shared_c[idx] != UInt32(0)
                @atomic output[NB + bin, b] += Float32(shared_c[idx])
            end
        end
    end
    @synchronize()
end

function _launch_v1_host_relaunch!(backend, output, u_dev, N, B, P, W, nworkers, wg)
    n_launches = 0
    t = @elapsed begin
        for pass_id in 1:P
            b_base = 1
            while b_base <= B
                bw = min(W, B - b_base + 1)
                kernel! = _toy_v1_host_relaunch!(backend, wg)
                kernel!(
                    output, u_dev, N, TOY_NB, b_base, bw, pass_id, nworkers;
                    ndrange = nworkers, workgroupsize = wg,
                )
                n_launches += 1
                b_base += W
            end
        end
        KA.synchronize(backend)
    end
    return t, n_launches
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function _backend()
    force = lowercase(strip(get(ENV, "BACKEND", "")))
    if force == "cpu"
        return KA.CPU(), "CPU"
    end
    if force == "cuda"
        haskey(ENV, "SLURM_JOB_ID") || get(ENV, "ALLOW_CUDA_BENCH", "") == "1" ||
            error("BACKEND=cuda requires SLURM_JOB_ID or ALLOW_CUDA_BENCH=1")
        CUDA.functional() || error("CUDA not functional")
        return CUDA.CUDABackend(), "CUDA"
    end
    if haskey(ENV, "SLURM_JOB_ID") && CUDA.functional()
        return CUDA.CUDABackend(), "CUDA"
    end
    if get(ENV, "ALLOW_CUDA_BENCH", "") == "1" && CUDA.functional()
        return CUDA.CUDABackend(), "CUDA"
    end
    return KA.CPU(), "CPU"
end

function _reset_output!(backend, B)
    out = zeros(Float32, 2 * TOY_NB, B)
    return KA.adapt(backend, out)
end

function _make_u(backend, N, B)
    u = rand(Float32, N, B)
    return KA.adapt(backend, u)
end

function _checksum(output)
    a = Array(output)
    return (sum(a), sum(abs.(a)))
end

function _parity_ok(a, b; rtol::Float64 = 1e-5, atol::Float64 = 64.0)
    return isapprox(Array(a), Array(b); rtol = rtol, atol = atol)
end

function _parity_report(a, b)
    da = Array(a)
    db = Array(b)
    d = abs.(da .- db)
    return (maximum(d), abs(sum(da) - sum(db)) / max(abs(sum(db)), 1.0))
end

function main()
    N = parse(Int, get(ENV, "N", "256"))
    B = parse(Int, get(ENV, "B", "8000"))
    P = parse(Int, get(ENV, "P", "4"))
    W = parse(Int, get(ENV, "STRIP_W", "32"))
    wg = parse(Int, get(ENV, "WORKGROUP", "256"))
    show_v1 = get(ENV, "SHOW_V1", "") == "1"

    W <= TOY_MAX_STRIP || error("STRIP_W=$W exceeds TOY_MAX_STRIP=$TOY_MAX_STRIP")

    backend, backend_name = _backend()
    nworkers = parse(Int, get(ENV, "NWORKERS", string(min(262_144, N * (N - 1) ÷ 2))))

    total_pairs = N * (N - 1) ÷ 2
    updates = total_pairs * B * P
    n_warmup = haskey(ENV, "WARMUP") ? parse(Int, ENV["WARMUP"]) : (updates > 1_000_000_000 ? 0 : 1)
    n_repeat = haskey(ENV, "N_REPEAT") ? parse(Int, ENV["N_REPEAT"]) : (updates > 1_000_000_000 ? 1 : 3)
    n_strip_chunks = P * cld(B, W)
    nblocks = _nblocks(nworkers, wg)
    priv_bytes = _private_partial_bytes(B, nblocks)
    slab_bytes = _private_slab_bytes(B)

    println("=== batch accum toy (one pair walk: VRAM vs private VRAM vs block smem) ===")
    println("backend=$backend_name  N=$N  B=$B  P=$P  strip_W=$W  NB=$TOY_NB")
    println("pairs=$total_pairs  updates(pair×b×p)=$updates  nblocks=$nblocks  wg=$wg")
    println("warmup=$n_warmup  n_repeat=$n_repeat  (override with WARMUP= / N_REPEAT=)")
    println("fused_vram:          1 accum launch   pair_traversals=$total_pairs  (global atomics)")
    println("fused_vram_private:  1 accum + merge  pair_traversals=$total_pairs")
    @printf("  partial total=%.3f GiB  per-block slab=%.2f MiB  (2×NB×B floats)\n",
        priv_bytes / (1024^3), slab_bytes / (1024^2))
    println("fused_block_smem:    1 launch         pair_traversals=$(total_pairs * n_strip_chunks)  (P×ceil(B/W) inner chunks)")
    println("NOTE: no tiled128 — absolute seconds are not production; smem arm replays pairs per strip.")
    show_v1 && println("v1_host_relaunch:    $(n_strip_chunks) launches  [rejected, SHOW_V1=1]")
    flush(stdout)

    u_dev = _make_u(backend, N, B)
    partial_dev = _make_partial!(backend, B, nblocks)

    for i in 1:n_warmup
        n_warmup > 0 && @printf("warmup %d/%d ...\n", i, n_warmup)
        flush(stdout)
        _launch_fused_vram!(backend, _reset_output!(backend, B), u_dev, N, B, P, nworkers, wg)
        _launch_fused_vram_private!(backend, _reset_output!(backend, B), partial_dev, u_dev, N, B, P, nworkers, wg)
        _launch_fused_block_smem!(backend, _reset_output!(backend, B), u_dev, N, B, P, W, nworkers, wg)
    end

    times_vram = Float64[]
    times_priv = Float64[]
    times_smem = Float64[]
    for i in 1:n_repeat
        @printf("timed run %d/%d: fused_vram ...\n", i, n_repeat)
        flush(stdout)
        out_v = _reset_output!(backend, B)
        push!(times_vram, _launch_fused_vram!(backend, out_v, u_dev, N, B, P, nworkers, wg))
        @printf("timed run %d/%d: fused_vram_private ...\n", i, n_repeat)
        flush(stdout)
        out_p = _reset_output!(backend, B)
        push!(times_priv, _launch_fused_vram_private!(backend, out_p, partial_dev, u_dev, N, B, P, nworkers, wg))
        @printf("timed run %d/%d: fused_block_smem ...\n", i, n_repeat)
        flush(stdout)
        out_s = _reset_output!(backend, B)
        push!(times_smem, _launch_fused_block_smem!(backend, out_s, u_dev, N, B, P, W, nworkers, wg))
    end

    t_vram = minimum(times_vram)
    t_priv = minimum(times_priv)
    t_smem = minimum(times_smem)

    out_v = _reset_output!(backend, B)
    _launch_fused_vram!(backend, out_v, u_dev, N, B, P, nworkers, wg)
    out_p = _reset_output!(backend, B)
    _launch_fused_vram_private!(backend, out_p, partial_dev, u_dev, N, B, P, nworkers, wg)
    out_s = _reset_output!(backend, B)
    _launch_fused_block_smem!(backend, out_s, u_dev, N, B, P, W, nworkers, wg)
    parity_vp = _parity_ok(out_v, out_p)
    parity_vs = _parity_ok(out_v, out_s)
    maxabs_p, relsum_p = _parity_report(out_v, out_p)
    maxabs_s, relsum_s = _parity_report(out_v, out_s)
    chk_v = _checksum(out_v)
    chk_p = _checksum(out_p)
    chk_s = _checksum(out_s)

    ups = updates / t_vram
    @printf("fused_vram          min=%.4fs  updates/s=%.3e\n", t_vram, ups)
    @printf("fused_vram_private  min=%.4fs  private/vram=%.3fx\n", t_priv, t_priv / t_vram)
    @printf("fused_block_smem    min=%.4fs  smem/vram=%.3fx\n", t_smem, t_smem / t_vram)
    @printf("checksum vram=(%.4g, %.4g)  private=(%.4g, %.4g)  smem=(%.4g, %.4g)\n",
        chk_v[1], chk_v[2], chk_p[1], chk_p[2], chk_s[1], chk_s[2])
    @printf("parity vram vs private=%s  maxabs=%.4g  rel_sum_diff=%.3e\n",
        parity_vp ? "PASS" : "FAIL", maxabs_p, relsum_p)
    @printf("parity vram vs smem=%s  maxabs=%.4g  rel_sum_diff=%.3e\n",
        parity_vs ? "PASS" : "FAIL", maxabs_s, relsum_s)
    parity_vp && parity_vs || error("checksum parity failed")

    if show_v1
        t_v1, n_v1 = _launch_v1_host_relaunch!(
            backend, _reset_output!(backend, B), u_dev, N, B, P, W, nworkers, wg,
        )
        @printf("v1_host_relaunch  min=%.4fs  host_launches=%d  (same pair work as fused_block_smem, extra launch tax)\n",
            t_v1, n_v1)
        @printf("v1 / fused_vram = %.3fx\n", t_v1 / t_vram)
    end

    println("done")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
