using StructureFunctions
using OhMyThreads
using Profile
using Serialization
using Random
using Printf

function main()
    profile_dir = @__DIR__
    
    if Threads.nthreads() < 24
        @warn "Current thread count is $(Threads.nthreads()). For standard profiling, please run Julia with 24 threads: `julia -t 24 --project=benchmark benchmark/profile/profile_runner.jl`"
    else
        println("Starting profiling runner with $(Threads.nthreads()) threads.")
        flush(stdout); flush(stderr)
    end

    # 1. Setup reproducible random data
    Random.seed!(42)
    N = 30000
    FT = Float64
    println("Generating 2D dataset with N = $N points...")
    flush(stdout); flush(stderr)
    x = randn(FT, 2, N)
    u = randn(FT, 2, N)

    # 50 log-spaced bin edges (length 51)
    distance_bins_raw = collect(exp.(range(log(0.01), log(10.0), length=51)))
    distance_bins = LogBinEdges(distance_bins_raw)

    # 2. Warmup run (excludes compilation time from profile)
    println("Running warmup calculations to trigger JIT compilation...")
    flush(stdout); flush(stderr)
    
    # warm up serial
    calculate_structure_functions_single_pass(x[:, 1:200], u[:, 1:200], distance_bins; backend=SerialBackend())
    # warm up threaded
    calculate_structure_functions_single_pass(x[:, 1:200], u[:, 1:200], distance_bins; backend=ThreadedBackend())

    # 3. CPU Profiling (Serial)
    println("Profiling CPU execution (Serial)...")
    flush(stdout); flush(stderr)

    Profile.clear()
    Profile.@profile calculate_structure_functions_single_pass(x, u, distance_bins; backend=SerialBackend())
    
    cpu_serial_txt = joinpath(profile_dir, "cpu_serial.txt")
    cpu_serial_jls = joinpath(profile_dir, "cpu_serial.jls")
    open(cpu_serial_txt, "w") do io
        Profile.print(io, format=:flat, sortedby=:rec)
    end
    serialize(cpu_serial_jls, Profile.retrieve())
    println("Saved Serial CPU Profile to:")
    println("  - Text format: $cpu_serial_txt")
    println("  - Binary data: $cpu_serial_jls")
    flush(stdout); flush(stderr)

    # 4. CPU Profiling (Threaded)
    println("Profiling CPU execution (Threaded)...")
    flush(stdout); flush(stderr)

    Profile.clear()
    Profile.@profile calculate_structure_functions_single_pass(x, u, distance_bins; backend=ThreadedBackend())
    
    cpu_threaded_txt = joinpath(profile_dir, "cpu_threaded.txt")
    cpu_threaded_jls = joinpath(profile_dir, "cpu_threaded.jls")
    open(cpu_threaded_txt, "w") do io
        Profile.print(io, format=:flat, sortedby=:rec)
    end
    serialize(cpu_threaded_jls, Profile.retrieve())
    println("Saved Threaded CPU Profile to:")
    println("  - Text format: $cpu_threaded_txt")
    println("  - Binary data: $cpu_threaded_jls")
    flush(stdout); flush(stderr)

    # 5. Allocations Profiling
    # Helper to count/group Allocations
    function write_allocs_summary(filename, alloc_profile)
        allocs = alloc_profile.allocs
        if isempty(allocs)
            open(filename, "w") do io
                println(io, "No allocations captured.")
            end
            return
        end

        open(filename, "w") do io
            println(io, "=== Allocations Summary ===")
            println(io, "Total allocations tracked: ", length(allocs))
            total_bytes = sum(a -> a.size, allocs)
            println(io, "Total tracked bytes: ", total_bytes)
            println(io)

            # Group by type
            by_type = Dict{Any, Vector{Int}}()
            for a in allocs
                stats = get!(by_type, a.type, [0, 0])
                stats[1] += 1
                stats[2] += a.size
            end

            println(io, "--- Top Types by Allocated Bytes ---")
            sorted_types = sort(collect(by_type), by = pair -> pair[2][2], rev=true)
            for (t, stats) in sorted_types[1:min(end, 20)]
                @printf(io, "  %s: %d allocations, %d bytes\n", string(t), stats[1], stats[2])
            end
            println(io)

            # Group by stacktrace location
            by_loc = Dict{String, Vector{Int}}()
            for a in allocs
                if !isempty(a.stacktrace)
                    sf = a.stacktrace[1]
                    loc = "$(sf.file):$(sf.line) ($(sf.func))"
                    stats = get!(by_loc, loc, [0, 0])
                    stats[1] += 1
                    stats[2] += a.size
                end
            end

            println(io, "--- Top Allocation Locations ---")
            sorted_locs = sort(collect(by_loc), by = pair -> pair[2][2], rev=true)
            for (loc, stats) in sorted_locs[1:min(end, 20)]
                @printf(io, "  %s: %d allocations, %d bytes\n", loc, stats[1], stats[2])
            end
        end
    end

    # Serial Allocations Profiling
    println("Profiling memory allocations (Serial)...")
    flush(stdout); flush(stderr)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.1 calculate_structure_functions_single_pass(x, u, distance_bins; backend=SerialBackend())
    allocs_serial_profile = Profile.Allocs.fetch()
    
    allocs_serial_txt = joinpath(profile_dir, "allocs_serial.txt")
    allocs_serial_jls = joinpath(profile_dir, "allocs_serial.jls")
    write_allocs_summary(allocs_serial_txt, allocs_serial_profile)
    serialize(allocs_serial_jls, allocs_serial_profile)
    println("Saved Serial Allocations Profile to:")
    println("  - Text format: $allocs_serial_txt")
    println("  - Binary data: $allocs_serial_jls")
    flush(stdout); flush(stderr)

    # Threaded Allocations Profiling
    println("Profiling memory allocations (Threaded)...")
    flush(stdout); flush(stderr)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.1 calculate_structure_functions_single_pass(x, u, distance_bins; backend=ThreadedBackend())
    allocs_threaded_profile = Profile.Allocs.fetch()
    
    allocs_threaded_txt = joinpath(profile_dir, "allocs_threaded.txt")
    allocs_threaded_jls = joinpath(profile_dir, "allocs_threaded.jls")
    write_allocs_summary(allocs_threaded_txt, allocs_threaded_profile)
    serialize(allocs_threaded_jls, allocs_threaded_profile)
    println("Saved Threaded Allocations Profile to:")
    println("  - Text format: $allocs_threaded_txt")
    println("  - Binary data: $allocs_threaded_jls")
    flush(stdout); flush(stderr)

    println("All profiling completed successfully.")
end

main()
