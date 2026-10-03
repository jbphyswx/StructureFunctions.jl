# Pseudo spherical-harmonic coefficients on a device, `C[l+1, m+lmax+1] = Σ_i f_i N_l d^l_{m,−s}(θ_i) e^{−imφ_i}`.

const HARM_WG = 128
const HARM_PTS = 2
const HARM_LT = 4
const HARM_GROUPS_PER_SM = 8

"""Launch plan of `_harmonic_tile_kernel!`: workgroups of `WG` lanes, each lane carrying `PTS` points' recurrences,
degrees in tiles of `LT`, each order's points split among `n_chunks` workgroups. A lane contributes `4·LT` values per
tile (the real and imaginary parts at `m` and, at spin 0, at `−m`), and `WG` is a multiple of them."""
struct HarmPlan{WG, PTS, LT}
    n_chunks::Int
end

"""The plan for `n_m` orders over `N` points on the device `caps` describes: `HARM_WG` lanes of `HARM_PTS` points,
degree tiles of `HARM_LT`, and chunks enough for `HARM_GROUPS_PER_SM` workgroups per multiprocessor — the constants with
the least worst-case regret over the regimes of the `harmonic` section of `gpu/benchmark_launch_plans.jl`."""
_harm_plan(caps::SFC.GPUDeviceCaps, n_m::Int, N::Int) =
    HarmPlan{HARM_WG, HARM_PTS, HARM_LT}(clamp(cld(HARM_GROUPS_PER_SM * caps.n_sms, n_m), 1,
                                               max(1, cld(N, HARM_WG * HARM_PTS))))

"""The coefficients `(a, c1, 1/c2)` of the recurrence step from degree `l` to `l + 1` of `d^l_{mn}`, those of
[`SFC.wigner_d_column!`](@ref)."""
@inline function _wigner_step(l::Int, m::Int, n::Int)
    a = l == 0 ? 0.0 : (m * n) / (l * (l + 1))
    c1 = l == 0 ? 0.0 : sqrt(Float64((l^2 - m^2) * (l^2 - n^2))) / (l * (2l + 1))
    c2 = sqrt(Float64(((l + 1)^2 - m^2) * ((l + 1)^2 - n^2))) / ((l + 1) * (2l + 1))
    return a, c1, inv(c2)
end

"""Order `m`, lowest degree, order index and point chunk of workgroup `grp`."""
@inline function _harm_group(grp::Int, n_chunks::Int, m_first::Int, s::Int)
    mi = (grp - 1) ÷ n_chunks
    chunk = (grp - 1) - mi * n_chunks
    m = m_first + mi
    return m, max(abs(m), abs(s)), mi, chunk
end

# A workgroup per (order m, point chunk). Each lane carries PTS points' Wigner recurrences and walks the degrees
# in tiles of LT: the tile's recurrence coefficients are shared, each lane sums its points' contributions in
# registers, the lanes' sums reduce in shared memory, and the workgroup adds each degree's total to its own slice
# of `scratch`. At spin 0 (`PAIR`) one recurrence serves `±m`, since d^l_{−m,0} = (−1)^m d^l_{m,0}.
KA.@kernel unsafe_indices = true function _harmonic_tile_kernel!(
    scratch,                # (4, lmax+1, n_m, n_chunks): re and im at m, re and im at −m
    @Const(f), @Const(θ), @Const(φ), @Const(lf),
    N::Int, lmax::Int, s::Int, m_first::Int, n_chunks::Int, n_batches::Int, n_tiles::Int, ::Val{PAIR},
    ::Val{WG}, ::Val{PTS}, ::Val{LT},
) where {PAIR, WG, PTS, LT}
    red = @localmem Float64 ((4 * LT + 1) * WG,)
    part = @localmem Float64 (WG,)
    coef = @localmem Float64 (4 * LT,)
    xs = @private Float64 (PTS,)
    prv = @private Float64 (PTS,)
    cur = @private Float64 (PTS,)
    ph = @private Float64 (4 * PTS,)
    acc = @private Float64 (4 * LT,)
    lid = @index(Local, Linear)
    grp = @index(Group, Linear)
    for batch in 1:n_batches
        m, l0, _, chunk = _harm_group(Int(grp), n_chunks, m_first, s)
        base = ((batch - 1) * n_chunks + chunk) * WG * PTS
        for k in 1:PTS
            i = base + Int(lid) + (k - 1) * WG
            if i <= N
                β = @inbounds θ[i]
                ϕ = @inbounds φ[i]
                fr, fi = reim(@inbounds f[i])
                cp, sp = cos(m * ϕ), sin(m * ϕ)
                sg = iseven(m) ? 1.0 : -1.0
                xs[k] = cos(β)
                prv[k] = 0.0
                cur[k] = SFC._wigner_seed(l0, m, -s, β, lf)
                ph[4k - 3] = fr * cp + fi * sp
                ph[4k - 2] = fi * cp - fr * sp
                ph[4k - 1] = sg * (fr * cp - fi * sp)
                ph[4k] = sg * (fi * cp + fr * sp)
            else
                xs[k] = 0.0
                prv[k] = 0.0
                cur[k] = 0.0
                ph[4k - 3] = 0.0
                ph[4k - 2] = 0.0
                ph[4k - 1] = 0.0
                ph[4k] = 0.0
            end
        end
        for t in 1:n_tiles
            m, l0, _, _ = _harm_group(Int(grp), n_chunks, m_first, s)
            lt = (t - 1) * LT
            lane = Int(lid)
            if lane <= LT
                l = lt + lane - 1
                a, c1, ic2 = (l0 <= l < lmax) ? _wigner_step(l, m, -s) : (0.0, 0.0, 0.0)
                @inbounds coef[4lane - 3] = a
                @inbounds coef[4lane - 2] = c1
                @inbounds coef[4lane - 1] = ic2
                @inbounds coef[4lane] = sqrt((2l + 1) / (4π))
            end
            @synchronize
            m, l0, _, _ = _harm_group(Int(grp), n_chunks, m_first, s)
            lt = (t - 1) * LT
            for c in 1:(4 * LT)
                acc[c] = 0.0
            end
            for k in 1:PTS, q in 1:LT
                l = lt + q - 1
                if l0 <= l <= lmax
                    v = cur[k] * @inbounds(coef[4q])
                    acc[q] += ph[4k - 3] * v
                    acc[LT + q] += ph[4k - 2] * v
                    if PAIR
                        acc[2 * LT + q] += ph[4k - 1] * v
                        acc[3 * LT + q] += ph[4k] * v
                    end
                    nxt = ((xs[k] - @inbounds(coef[4q - 3])) * cur[k] - @inbounds(coef[4q - 2]) * prv[k]) *
                          @inbounds(coef[4q - 1])
                    prv[k] = cur[k]
                    cur[k] = nxt
                end
            end
            lane = Int(lid)
            for c in 1:(4 * LT)
                @inbounds red[(lane - 1) * (4 * LT + 1) + c] = acc[c]
            end
            @synchronize
            lane = Int(lid)
            val = (lane - 1) % (4 * LT) + 1
            g = (lane - 1) ÷ (4 * LT)
            s1 = 0.0
            for j in 1:(4 * LT)
                s1 += @inbounds red[(g * 4 * LT + j - 1) * (4 * LT + 1) + val]
            end
            @inbounds part[g * 4 * LT + val] = s1
            @synchronize
            m, l0, mi, chunk = _harm_group(Int(grp), n_chunks, m_first, s)
            lt = (t - 1) * LT
            lane = Int(lid)
            if lane <= 4 * LT
                c = (lane - 1) ÷ LT + 1
                q = (lane - 1) % LT + 1
                l = lt + q - 1
                if l0 <= l <= lmax && (PAIR || c <= 2)
                    tot = 0.0
                    for g2 in 0:(WG ÷ (4 * LT) - 1)
                        tot += @inbounds part[g2 * 4 * LT + lane]
                    end
                    @inbounds scratch[c, l + 1, mi + 1, chunk + 1] += tot
                end
            end
            @synchronize
        end
    end
end

"""`a` on the device of `ka` with element type `T`, converting only when it has another."""
_harm_array(ka, ::Type{T}, a) where {T} = (d = KA.adapt(ka, a); eltype(d) === T ? d : T.(d))

SFC._direct_coefficients(gb::CB.AbstractGPUBackend, f, θ, φ, s, lmax) =
    _harm_coefficients(_harm_plan, gb.backend, f, θ, φ, s, lmax)

"""The device pseudo-coefficients of [`SFC._direct_coefficients`](@ref) with the plan `choose(caps, n_m, N)`
returns (see [`_harm_plan`](@ref))."""
function _harm_coefficients(choose::C, ka, f, θ, φ, s, lmax) where {C}
    L, S, N = Int(lmax), Int(s), length(f)
    pair = S == 0
    m_first, n_m = pair ? (0, L + 1) : (-L, 2L + 1)
    hp = choose(SFC.gpu_device_caps(ka), n_m, N)
    scratch = KA.zeros(ka, Float64, 4, L + 1, n_m, hp.n_chunks)
    _harm_launch!(hp, ka, scratch, _harm_array(ka, ComplexF64, f), _harm_array(ka, Float64, θ),
                  _harm_array(ka, Float64, φ), KA.adapt(ka, SFC._log_factorials(2L + 2)), N, L, S, m_first, n_m,
                  Val(pair))
    tot = dropdims(sum(scratch; dims = 4); dims = 4)
    at_m = complex.(view(tot, 1, :, :), view(tot, 2, :, :))
    pair || return at_m
    at_neg = complex.(view(tot, 3, :, 2:n_m), view(tot, 4, :, 2:n_m))
    return hcat(at_neg[:, end:-1:1], at_m)
end

"""Launch `_harmonic_tile_kernel!` with the plan `hp` over `n_m` orders from `m_first`, into `scratch`."""
function _harm_launch!(hp::HarmPlan{WG, PTS, LT}, ka, scratch, f, θ, φ, lf, N::Int, L::Int, S::Int, m_first::Int,
                       n_m::Int, vpair::Val) where {WG, PTS, LT}
    _harmonic_tile_kernel!(ka, WG)(scratch, f, θ, φ, lf, N, L, S, m_first, hp.n_chunks, cld(N, hp.n_chunks * WG * PTS),
                                   cld(L + 1, LT), vpair, Val(WG), Val(PTS), Val(LT); ndrange = n_m * hp.n_chunks * WG)
    return nothing
end
