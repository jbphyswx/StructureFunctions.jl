using Serialization
using Printf
using Profile

function generate_cpu_summary(input_file, output_file)
    if !isfile(input_file)
        println("Input file $input_file not found.")
        return
    end
    
    data, lkup = open(deserialize, input_file)
    
    # Reconstruct backtraces
    counts = Dict{String, Int}()
    total_samples = 0
    
    current_bt = UInt64[]
    for ip in data
        if ip == 0
            if !isempty(current_bt)
                is_idle = false
                for tip in current_bt
                    frames = get(lkup, tip, nothing)
                    if !isnothing(frames)
                        for frame in frames
                            func = string(frame.func)
                            file = string(frame.file)
                            if func in ("pthread_cond_wait", "uv_cond_wait", "__futex_abstimed_wait_common", "jl_parallel_gc_threadfun", "poptask")
                                is_idle = true
                                break
                            end
                        end
                    end
                    if is_idle; break; end
                end
                
                if !is_idle
                    seen = Set{String}()
                    for tip in current_bt
                        frames = get(lkup, tip, nothing)
                        if !isnothing(frames)
                            for frame in frames
                                func = string(frame.func)
                                file = string(frame.file)
                                # Skip base runtime internals to keep it highly readable
                                if occursin("task.jl", file) || occursin("boot.jl", file) || occursin("client.jl", file) || occursin("loading.jl", file) || occursin("compiler", file) || occursin("gc-alloc", file)
                                    continue
                                end
                                loc = "$(func) at $(basename(file)):$(frame.line)"
                                push!(seen, loc)
                            end
                        end
                    end
                    for loc in seen
                        counts[loc] = get(counts, loc, 0) + 1
                    end
                    total_samples += 1
                end
                current_bt = UInt64[]
            end
        else
            push!(current_bt, ip)
        end
    end

    # Sort descending
    sorted_counts = sort(collect(counts), by = pair -> pair[2], rev = true)
    
    open(output_file, "w") do io
        println(io, "=== CPU PROFILE SUMMARY ===")
        println(io, "File: ", basename(input_file))
        println(io, "Total snapshots: ", total_samples)
        println(io)
        @printf(io, "%-8s  %s\n", "Cost %", "Function / Location")
        println(io, "-"^80)
        for (loc, count) in sorted_counts
            pct = (count / total_samples) * 100
            if pct >= 0.2 # Filter out noise below 0.2%
                @printf(io, "%6.2f%%   %s\n", pct, loc)
            end
        end
    end
    println("Generated summary: $output_file")
end

function generate_all_readouts()
    profile_dir = @__DIR__
    # CPU summaries
    generate_cpu_summary(joinpath(profile_dir, "cpu_serial.jls"), joinpath(profile_dir, "cpu_serial_summary.txt"))
    generate_cpu_summary(joinpath(profile_dir, "cpu_threaded.jls"), joinpath(profile_dir, "cpu_threaded_summary.txt"))
end

generate_all_readouts()
