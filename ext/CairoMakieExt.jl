module CairoMakieExt

using CairoMakie: CairoMakie
import StructureFunctions.SpectralAnalysis: plot_spectrum, compare_spectra

"""
    plot_spectrum(k_ranges::Tuple, coeffs::Array; 
        title="Power Spectrum", 
        xlabel="kx", 
        ylabel="ky",
        u_idx=1
    )

Plot the magnitude of the spectral coefficients.
"""
function plot_spectrum(k_ranges::Tuple, coeffs::Array; 
    title="Power Spectrum", 
    u_idx=1,
    kwargs...
)
    # Handle 1D
    if length(k_ranges) == 1
        k = k_ranges[1]
        c = abs.(selectdim(coeffs, 2, u_idx))
        
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1]; title=title, xlabel="k", ylabel="|c(k)|", kwargs...)
        CairoMakie.lines!(ax, k, c)
        return fig
    end
    
    # Handle 2D
    if length(k_ranges) == 2
        kx, ky = k_ranges
        c = abs.(selectdim(coeffs, 3, u_idx))
        
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1]; title=title, xlabel="kx", ylabel="ky", kwargs...)
        hm = CairoMakie.heatmap!(ax, kx, ky, c)
        CairoMakie.Colorbar(fig[1, 2], hm)
        return fig
    end
    
    error("plot_spectrum only supports 1D and 2D results currently.")
end

"""
    compare_spectra(results; peaks=nothing, u_idx=1, title="Spectral Analysis Comparison", kwargs...)

Plot multiple spectra side-by-side and optionally mark expected peaks.
"""
function compare_spectra(results; peaks=nothing, u_idx=1, title="Spectral Analysis Comparison", kwargs...)
    N = length(results)
    # 0.15+ uses size, resolution is deprecated
    fig = CairoMakie.Figure(; size=(600 * N, 550))
    CairoMakie.Label(fig[0, :], title, fontsize=20, font=:bold)
    
    for (i, (label, (coeffs, k_ranges))) in enumerate(results)
        D = length(k_ranges)
        ax = CairoMakie.Axis(fig[1, i]; title=label, xlabel=D==1 ? "k" : "kx", ylabel=D==1 ? "|c(k)|" : "ky")
        
        if D == 1
            CairoMakie.lines!(ax, k_ranges[1], abs.(selectdim(coeffs, 2, u_idx)))
            
            # Mark peaks if provided
            if peaks !== nothing
                for p in peaks
                    # p is (k1, k2, amp) or similar
                    kp = p[1]
                    CairoMakie.vlines!(ax, [kp], color=:red, linestyle=:dash, alpha=0.5)
                end
            end
        else
            # Heatmap expects x, y, z
            hm = CairoMakie.heatmap!(ax, k_ranges[1], k_ranges[2], abs.(selectdim(coeffs, 3, u_idx)))
            
            # Mark peaks if provided
            if peaks !== nothing
                p_kx = [p[1] for p in peaks]
                p_ky = [p[2] for p in peaks]
                CairoMakie.scatter!(ax, p_kx, p_ky, color=:red, marker=:xcross, markersize=15, label="Target Peaks")
            end
            
            if i == N
                CairoMakie.Colorbar(fig[1, i+1], hm)
            end
        end
    end
    
    return fig
end

end # module
