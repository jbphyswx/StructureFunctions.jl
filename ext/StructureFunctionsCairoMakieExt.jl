module StructureFunctionsCairoMakieExt

using CairoMakie: CairoMakie
using StructureFunctions: StructureFunctions as SF, SpectralAnalysis as SFSA

"""
    plot_spectrum(k_ranges::Tuple, coeffs::Array; 
        title="Power Spectrum", 
        xlabel="kx", 
        ylabel="ky",
        u_idx=1
    )

Plot the magnitude of the spectral coefficients.
"""
function SFSA.plot_spectrum(k_ranges::Tuple, coeffs::Array; 
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
function SFSA.compare_spectra(results; peaks=nothing, u_idx=1, title="Spectral Analysis Comparison", kwargs...)
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

"""
    compare_spectral_analysis(input_data, results; peaks=nothing, u_idx=1, title="Spectral Analysis Comparison", kwargs...)

Prepend a plot of the input field to the spectral comparison.
input_data: Tuple(x_vecs, u_vecs)
"""
function SFSA.compare_spectral_analysis(input_data, results; peaks=nothing, u_idx=1, title="Spectral Analysis Comparison", kwargs...)
    x_vecs, u_vecs = input_data
    NU = length(u_vecs)
    D = length(x_vecs)
    N_results = length(results)
    
    # Grid: 1 field plot + N spectral plots
    fig = CairoMakie.Figure(; size=(600 * (N_results + 1), 550))
    CairoMakie.Label(fig[0, :], title, fontsize=20, font=:bold)
    
    # 1. Plot Input Field
    ax_field = CairoMakie.Axis(fig[1, 1]; title="Input Field (u[$u_idx])", xlabel=D==1 ? "x" : "x", ylabel=D==1 ? "u" : "y")
    u_plot = u_vecs[u_idx]
    
    if D == 1
        CairoMakie.lines!(ax_field, x_vecs[1], real.(u_plot))
    else
        # Check if approx uniform to use heatmap, else scatter
        # For simplicity in tests, we can use scatter if it's large, but heatmap is better for high-res
        # Let's assume scatter for now as these are often non-uniform in our tests
        CairoMakie.scatter!(ax_field, x_vecs[1], x_vecs[2], color=real.(u_plot), markersize=4)
    end
    
    # 2. Spectral Results (shifted by 1)
    for (i, (label, (coeffs, k_ranges))) in enumerate(results)
        spectral_idx = i + 1
        ax = CairoMakie.Axis(fig[1, spectral_idx]; title=label, xlabel=D==1 ? "k" : "kx", ylabel=D==1 ? "|c(k)|" : "ky")
        
        if D == 1
            CairoMakie.lines!(ax, k_ranges[1], abs.(Base.selectdim(coeffs, 2, u_idx)))
            if peaks !== nothing
                for p in peaks
                    CairoMakie.vlines!(ax, [p[1]], color=:red, linestyle=:dash, alpha=0.5)
                end
            end
        else
            hm = CairoMakie.heatmap!(ax, k_ranges[1], k_ranges[2], abs.(Base.selectdim(coeffs, 3, u_idx)))
            if peaks !== nothing
                p_kx = [p[1] for p in peaks]
                p_ky = [p[2] for p in peaks]
                CairoMakie.scatter!(ax, p_kx, p_ky, color=:red, marker=:xcross, markersize=15, label="Target Peaks")
            end
            if i == N_results
                CairoMakie.Colorbar(fig[1, spectral_idx+1], hm)
            end
        end
    end
    
    return fig
end

end # module
