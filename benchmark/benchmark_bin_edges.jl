using BenchmarkTools
using StructureFunctions: StructureFunctions as SF
using Random

function run_bin_edge_benchmarks()
    Random.seed!(42)
    N_queries = 10_000
    queries = rand(N_queries) .* 15.0

    println("=========================================================")
    # 1. Linear Bins Benchmarks (Uniform Spacing)
    # =========================================================
    n_lin_bins = 1000
    lin_range = range(0.01, 10.0, length=n_lin_bins)
    lin_vec = collect(lin_range)
    lin_be = SF.LinearBinEdges(lin_range)

    println("--- 1. Linear Bin Edge Lookup (N = $n_lin_bins) ---")
    
    println("Standard Vector Binary Search:")
    t_vec = @belapsed (s = 0; for q in $queries; s += searchsortedfirst($lin_vec, q); end; s)
    println("  Time per query: ", (t_vec / N_queries) * 1e9, " ns")

    println("LinearBinEdges (O(1) FMA Search):")
    t_be = @belapsed (s = 0; for q in $queries; s += searchsortedfirst($lin_be, q); end; s)
    println("  Time per query: ", (t_be / N_queries) * 1e9, " ns")
    println("  Speedup: ", round(t_vec / t_be, digits=1), "x")
    println()

    # =========================================================
    # 2. Log-spaced Bins Benchmarks (Exponential Spacing)
    # =========================================================
    n_log_bins = 1000
    log_range = range(log(0.01), log(10.0), length=n_log_bins)
    log_vec = exp.(log_range)
    log_be = SF.LogBinEdges(log_vec)

    println("--- 2. Log-Spaced Bin Edge Lookup (N = $n_log_bins) ---")
    
    println("Standard Vector Binary Search:")
    t_log_vec = @belapsed (s = 0; for q in $queries; s += searchsortedfirst($log_vec, q); end; s)
    println("  Time per query: ", (t_log_vec / N_queries) * 1e9, " ns")

    println("LogBinEdges (O(1) Exponent LUT Hybrid Search):")
    t_log_be = @belapsed (s = 0; for q in $queries; s += searchsortedfirst($log_be, q); end; s)
    println("  Time per query: ", (t_log_be / N_queries) * 1e9, " ns")
    println("  Speedup: ", round(t_log_vec / t_log_be, digits=1), "x")
    println()

    # =========================================================
    # 3. Single-Pass Structure Function Benchmarks
    # =========================================================
    N_points = 2000
    x = randn(2, N_points)
    u = randn(2, N_points)

    println("--- 3. Single-Pass Structure Function Calculations (N_points = $N_points) ---")
    
    println("Using Standard Vector Bins (Binary Search):")
    b_standard = @benchmark SF.calculate_structure_functions_single_pass($x, $u, $log_vec; backend=SF.SerialBackend())
    display(b_standard)
    println()

    println("Using LogBinEdges Bins (O(1) Exponent LUT Search):")
    b_optimized = @benchmark SF.calculate_structure_functions_single_pass($x, $u, $log_be; backend=SF.SerialBackend())
    display(b_optimized)
    println()
end

run_bin_edge_benchmarks()
