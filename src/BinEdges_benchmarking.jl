using BenchmarkTools
using StaticArrays
using Printf

# Custom Types
abstract type AbstractBinEdges{T} <: AbstractVector{T} end

# 1. Custom FMA Linear Wrapper
struct LinearBinEdges{T, RT <: AbstractRange{T}} <: AbstractBinEdges{T}
    edges::RT
    inv_step::T
    offset::T
    first_edge::T
    last_edge::T
    step_val::T
end

function LinearBinEdges(edges::AbstractRange{T}) where {T}
    inv_step = inv(step(edges))
    offset = T(1.0) - first(edges) * inv_step
    return LinearBinEdges{T, typeof(edges)}(edges, inv_step, offset, first(edges), last(edges), step(edges))
end

Base.size(v::LinearBinEdges) = size(v.edges)
Base.getindex(v::LinearBinEdges, i::Int) = v.edges[i]

function Base.searchsortedfirst(v::LinearBinEdges{T}, x) where {T}
    f = v.first_edge
    if x <= f; return 1; end
    l = v.last_edge
    if x > l; return length(v.edges) + 1; end
    
    idx = round(Int, muladd(x, v.inv_step, v.offset))
    idx = clamp(idx, 1, length(v.edges))
    @inbounds edge_val = muladd(T(idx - 1), v.step_val, f)
    return edge_val < x ? idx + 1 : idx
end

# 2. Log wrappers
struct LogBinEdges{T, RT <: AbstractRange, VT <: AbstractVector} <: AbstractBinEdges{T}
    log_edges::RT
    edges::VT
    lut::Vector{Int}
    e_min::Int
    e_max::Int
    inv_step::T
    offset::T
end

function LogBinEdges(edges::AbstractVector{T}) where {T}
    log_start = log(first(edges))
    log_stop = log(last(edges))
    log_edges = range(log_start, log_stop, length=length(edges))
    
    e_min = exponent(first(edges))
    e_max = exponent(last(edges))
    lut_size = e_max - e_min + 2
    lut = Vector{Int}(undef, lut_size)
    for e in e_min:(e_max+1)
        val = T(2.0^e)
        lut[e - e_min + 1] = searchsortedfirst(edges, val)
    end
    
    inv_step = inv(step(log_edges))
    offset = T(1.0) - first(log_edges) * inv_step
    
    return LogBinEdges{T, typeof(log_edges), typeof(edges)}(log_edges, edges, lut, e_min, e_max, inv_step, offset)
end

Base.size(v::LogBinEdges) = size(v.edges)
Base.getindex(v::LogBinEdges, i::Int) = v.edges[i]

# Exponent LUT Search
function search_lut(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    e = exponent(x)
    e_idx = clamp(e - v.e_min + 1, 1, length(v.lut))
    idx = @inbounds v.lut[e_idx]
    
    @inbounds while idx <= length(v.edges) && v.edges[idx] < x
        idx += 1
    end
    return idx
end

# FMA Search (Ceil-based)
function search_fma(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    lx = log(x)
    return ceil(Int, muladd(lx, v.inv_step, v.offset))
end

# Ceil Search (Division-based)
function search_ceil_div(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    lx = log(x)
    return ceil(Int, (lx - first(v.log_edges)) / step(v.log_edges)) + 1
end

# Round Guess + Boundary Correction
function search_round_corrected(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    lx = log(x)
    g = round(Int, (lx - first(v.log_edges)) / step(v.log_edges) + 1)
    g = clamp(g, 1, length(v.edges))
    if x <= v.edges[g]
        if g > 1 && x <= v.edges[g-1]
            return g - 1
        else
            return g
        end
    else
        if g < length(v.edges) && x > v.edges[g+1]
            return g + 2
        else
            return g + 1
        end
    end
end

# Round Guess (No Check)
function search_round_nocheck(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    lx = log(x)
    g = round(Int, (lx - first(v.log_edges)) / step(v.log_edges) + 1)
    return clamp(g, 1, length(v.edges))
end

# Julia Base range search on logs
function search_julia_log_range(v::LogBinEdges, x)
    if x <= 0; return 1; end
    lx = log(x)
    return searchsortedfirst(v.log_edges, lx)
end

# FMA Ceil-based with 1-step Correction
function search_fma_corrected(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    lx = log(x)
    idx = ceil(Int, muladd(lx, v.inv_step, v.offset))
    idx = clamp(idx, 1, length(v.edges))
    @inbounds if idx > 1 && v.edges[idx-1] >= x
        return idx - 1
    elseif @inbounds idx <= length(v.edges) && v.edges[idx] < x
        return idx + 1
    end
    return idx
end

# Exponent LUT + Inline Binary Search within Octave Range
function search_lut_binary(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    e = exponent(x)
    e_idx = clamp(e - v.e_min + 1, 1, length(v.lut))
    idx_start = @inbounds v.lut[e_idx]
    idx_end = (e_idx < length(v.lut)) ? @inbounds(v.lut[e_idx+1]) : length(v.edges)
    
    low = idx_start
    high = idx_end
    @inbounds while low <= high
        mid = (low + high) >>> 1
        if v.edges[mid] < x
            low = mid + 1
        else
            high = mid - 1
        end
    end
    return low
end

# Exponent LUT + Hybrid Scan/Binary Search
function search_lut_hybrid(v::LogBinEdges, x)
    f = first(v.edges)
    if x <= f; return 1; end
    l = last(v.edges)
    if x > l; return length(v.edges) + 1; end
    
    e = exponent(x)
    e_idx = clamp(e - v.e_min + 1, 1, length(v.lut))
    idx_start = @inbounds v.lut[e_idx]
    idx_end = (e_idx < length(v.lut)) ? @inbounds(v.lut[e_idx+1]) : length(v.edges)
    
    if idx_end - idx_start <= 8
        idx = idx_start
        @inbounds while idx <= idx_end && v.edges[idx] < x
            idx += 1
        end
        return idx
    else
        low = idx_start
        high = idx_end
        @inbounds while low <= high
            mid = (low + high) >>> 1
            if v.edges[mid] < x
                low = mid + 1
            else
                high = mid - 1
            end
        end
        return low
    end
end

# Helper to run benchmark and return median time in ns
function run_bench(f, container, queries)
    # Warmup
    f(container, queries[1])
    # Benchmark
    t = @belapsed sum(q -> $f($container, q), $queries)
    return (t * 1e9) / length(queries) # time per query in ns
end

function run_benchmarks_for_type(::Type{T}) where {T}
    println("=" ^ 60)
    println("BENCHMARKS FOR TYPE: $T")
    println("=" ^ 60)
    flush(stdout); flush(stderr)
    
    # 1000 query points
    queries = rand(T, 1000) .* T(999.0) .+ T(1.0)
    
    for N in [51, 1000]
        println("-" ^ 40)
        println("SIZE N = $N")
        println("-" ^ 40)
        flush(stdout); flush(stderr)
        
        # Setup Linear Bins
        lin_range = range(T(1.0), T(1000.0), length=N)
        lin_vector = collect(lin_range)
        custom_lin = LinearBinEdges(lin_range)
        
        # Setup Log Bins
        log_range = range(log(T(1.0)), log(T(1000.0)), length=N)
        log_vector = exp.(log_range)
        log_vector_log = collect(log_range)
        custom_log = LogBinEdges(log_vector)
        
        # Create validation queries
        verify_queries = rand(T, 10000) .* T(999.0) .+ T(1.0)
        for i in 1:length(log_vector)
            push!(verify_queries, log_vector[i])
            push!(verify_queries, log_vector[i] - T(1e-5)) # coarse spacing for out of boundary noise
            push!(verify_queries, log_vector[i] + T(1e-5))
            push!(verify_queries, log_vector[i] - eps(log_vector[i]))
            push!(verify_queries, log_vector[i] + eps(log_vector[i]))
        end
        push!(verify_queries, T(0.5))
        push!(verify_queries, T(1005.0))
        
        # ------------------ LINEAR BINS ------------------
        println("--- LINEAR BINS ---")
        flush(stdout); flush(stderr)
        t_lin_bin = run_bench((v, x) -> searchsortedfirst(v, x), lin_vector, queries)
        t_lin_jul = run_bench((v, x) -> searchsortedfirst(v, x), lin_range, queries)
        t_lin_fma = run_bench((v, x) -> searchsortedfirst(v, x), custom_lin, queries)
        
        # Verify correctness for linear bins
        lin_errs_jul = 0
        lin_errs_fma = 0
        for q in verify_queries
            ref = searchsortedfirst(lin_vector, q)
            if searchsortedfirst(lin_range, q) != ref; lin_errs_jul += 1; end
            if searchsortedfirst(custom_lin, q) != ref; lin_errs_fma += 1; end
        end
        
        @printf("V1: Native Julia searchsortedfirst(::Vector): %6.2f ns (errors: 0)\n", t_lin_bin)
        flush(stdout); flush(stderr)
        @printf("V2: Native Julia searchsortedfirst(::StepRangeLen): %6.2f ns (errors: %d)\n", t_lin_jul, lin_errs_jul)
        flush(stdout); flush(stderr)
        @printf("V3: Custom FMA O(1) Search:                   %6.2f ns (errors: %d)\n", t_lin_fma, lin_errs_fma)
        flush(stdout); flush(stderr)
        println()
        
        # ------------------ LOG BINS ------------------
        println("--- LOG-SPACED BINS ---")
        t_log_bin = run_bench((v, x) -> searchsortedfirst(v, x), log_vector, queries)
        t_log_jul = run_bench(search_julia_log_range, custom_log, queries)
        t_log_bin_log = run_bench((v, x) -> searchsortedfirst(v, log(x)), log_vector_log, queries)
        t_log_round_no = run_bench(search_round_nocheck, custom_log, queries)
        t_log_round_cor = run_bench(search_round_corrected, custom_log, queries)
        t_log_ceil_div = run_bench(search_ceil_div, custom_log, queries)
        t_log_fma = run_bench(search_fma, custom_log, queries)
        t_log_fma_cor = run_bench(search_fma_corrected, custom_log, queries)
        t_log_lut = run_bench(search_lut, custom_log, queries)
        t_log_lut_bin = run_bench(search_lut_binary, custom_log, queries)
        t_log_lut_hyb = run_bench(search_lut_hybrid, custom_log, queries)
        
        # Helper to compute errors
        get_errs(f) = sum(q -> f(custom_log, q) != searchsortedfirst(log_vector, q) ? 1 : 0, verify_queries)
        
        err_jul = get_errs(search_julia_log_range)
        err_bin_log = sum(q -> searchsortedfirst(log_vector_log, log(q)) != searchsortedfirst(log_vector, q) ? 1 : 0, verify_queries)
        err_round_no = get_errs(search_round_nocheck)
        err_round_cor = get_errs(search_round_corrected)
        err_ceil_div = get_errs(search_ceil_div)
        err_fma = get_errs(search_fma)
        err_fma_cor = get_errs(search_fma_corrected)
        err_lut = get_errs(search_lut)
        err_lut_bin = get_errs(search_lut_binary)
        err_lut_hyb = get_errs(search_lut_hybrid)
        
        @printf("V1: Native Julia searchsortedfirst(::Vector): %6.2f ns (errors: 0)\n", t_log_bin)
        flush(stdout); flush(stderr)
        @printf("V2: Range searchsortedfirst on logs (Range):  %6.2f ns (errors: %d)\n", t_log_jul, err_jul)
        flush(stdout); flush(stderr)
        @printf("V2b: Log + Native searchsortedfirst(::Vector):%6.2f ns (errors: %d)\n", t_log_bin_log, err_bin_log)
        flush(stdout); flush(stderr)
        @printf("V3: Round Guess (No Check):                   %6.2f ns (errors: %d)\n", t_log_round_no, err_round_no)
        flush(stdout); flush(stderr)
        @printf("V4: Round Guess + Correction:                 %6.2f ns (errors: %d)\n", t_log_round_cor, err_round_cor)
        flush(stdout); flush(stderr)
        @printf("V5: Ceil Guess (No Check):                    %6.2f ns (errors: %d)\n", t_log_ceil_div, err_ceil_div)
        flush(stdout); flush(stderr)
        @printf("V6: FMA Ceil-based O(1):                      %6.2f ns (errors: %d)\n", t_log_fma, err_fma)
        flush(stdout); flush(stderr)
        @printf("V6b: FMA Ceil + ULP Correction:               %6.2f ns (errors: %d)\n", t_log_fma_cor, err_fma_cor)
        flush(stdout); flush(stderr)
        @printf("V7: Bitwise Exponent LUT:                     %6.2f ns (errors: %d)\n", t_log_lut, err_lut)
        flush(stdout); flush(stderr)
        @printf("V8: Exponent LUT + Sub-Binary:                %6.2f ns (errors: %d)\n", t_log_lut_bin, err_lut_bin)
        flush(stdout); flush(stderr)
        @printf("V9: Exponent LUT Hybrid:                      %6.2f ns (errors: %d)\n", t_log_lut_hyb, err_lut_hyb)
        flush(stdout); flush(stderr)
        println()
    end
end

function run_all_benchmarks()
    run_benchmarks_for_type(Float64)
    run_benchmarks_for_type(Float32)
    println("done.")
end

run_all_benchmarks()
