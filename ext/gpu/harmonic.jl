# Pseudo spherical-harmonic coefficients on a device.
#
# The sum is a reduction over points, so one thread owns one point and accumulates every `(l, m)`
# it touches with global atomics. Each thread needs its own Wigner column, which is `lmax + 1`
# long and therefore not a compile-time size; it lives in a global scratch matrix, one column per
# thread. Real and imaginary parts accumulate separately because an atomic add takes a real.

KA.@kernel unsafe_indices = true function _harmonic_direct_kernel!(
    acc,                    # (2, lmax+1, 2lmax+1): real and imaginary parts
    @Const(fre), @Const(fim), @Const(θ), @Const(φ),
    col,                    # (lmax+1, ndrange) Wigner scratch, one column per thread
    @Const(lf), @Const(norms),
    N_points::Int, lmax::Int, s::Int,
)
    i = @index(Global)
    if i <= N_points
        fr = @inbounds fre[i]
        fi = @inbounds fim[i]
        if !(iszero(fr) && iszero(fi))
            β = @inbounds θ[i]
            ϕ = @inbounds φ[i]
            mycol = @inbounds view(col, :, i)
            if s == 0
                # d^l_{-m,0} = (-1)^m d^l_{m,0}, so one recurrence serves both signs of m
                for m in 0:lmax
                    SFC.wigner_d_column!(mycol, m, 0, β, lmax, lf)
                    cp, sp = cos(m * ϕ), sin(m * ϕ)
                    # f * cis(-m φ)
                    pr = fr * cp + fi * sp
                    pi_ = fi * cp - fr * sp
                    # (-1)^m f * cis(+m φ)
                    sgn = iseven(m) ? 1.0 : -1.0
                    qr = sgn * (fr * cp - fi * sp)
                    qi = sgn * (fi * cp + fr * sp)
                    for l in m:lmax
                        v = @inbounds norms[l + 1] * mycol[l + 1]
                        @atomic acc[1, l + 1, m + lmax + 1] += pr * v
                        @atomic acc[2, l + 1, m + lmax + 1] += pi_ * v
                        if m != 0
                            @atomic acc[1, l + 1, lmax + 1 - m] += qr * v
                            @atomic acc[2, l + 1, lmax + 1 - m] += qi * v
                        end
                    end
                end
            else
                for m in -lmax:lmax
                    SFC.wigner_d_column!(mycol, m, -s, β, lmax, lf)
                    cp, sp = cos(m * ϕ), sin(m * ϕ)
                    pr = fr * cp + fi * sp
                    pi_ = fi * cp - fr * sp
                    lo = max(abs(m), abs(s))
                    for l in lo:lmax
                        v = @inbounds norms[l + 1] * mycol[l + 1]
                        @atomic acc[1, l + 1, m + lmax + 1] += pr * v
                        @atomic acc[2, l + 1, m + lmax + 1] += pi_ * v
                    end
                end
            end
        end
    end
end

function SFC._direct_coefficients(gb::CB.AbstractGPUBackend, f, θ, φ, s, lmax)
    ka = gb.backend
    N = length(f)
    L = Int(lmax)
    fc = ComplexF64.(f)
    acc_dev = KA.adapt(ka, zeros(Float64, 2, L + 1, 2L + 1))
    fre_dev = KA.adapt(ka, Float64.(real.(fc)))
    fim_dev = KA.adapt(ka, Float64.(imag.(fc)))
    θ_dev = KA.adapt(ka, Float64.(collect(θ)))
    φ_dev = KA.adapt(ka, Float64.(collect(φ)))
    col_dev = KA.adapt(ka, zeros(Float64, L + 1, N))
    lf_dev = KA.adapt(ka, SFC._log_factorials(2L + 2))
    norms_dev = KA.adapt(ka, [sqrt((2l + 1) / (4π)) for l in 0:L])

    kernel = _harmonic_direct_kernel!(ka, 64)
    kernel(acc_dev, fre_dev, fim_dev, θ_dev, φ_dev, col_dev, lf_dev, norms_dev,
           N, L, Int(s); ndrange = N)
    KA.synchronize(ka)

    acc = Array(acc_dev)
    out = Matrix{ComplexF64}(undef, L + 1, 2L + 1)
    @inbounds for m in 1:(2L + 1), l in 1:(L + 1)
        out[l, m] = complex(acc[1, l, m], acc[2, l, m])
    end
    return out
end
