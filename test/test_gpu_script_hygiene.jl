using Test: Test

function _script_files_under(paths)
    files = String[]
    for path in paths
        isfile(path) && endswith(path, ".jl") && push!(files, path)
        if isdir(path)
            for (root, _, names) in walkdir(path)
                for name in names
                    endswith(name, ".jl") || continue
                    push!(files, joinpath(root, name))
                end
            end
        end
    end
    return sort(files)
end

Test.@testset "GPU script hygiene" begin
    repo = normpath(joinpath(@__DIR__, ".."))
    paths = [
        joinpath(repo, "ext", "StructureFunctionsKernelAbstractionsExt.jl"),
        joinpath(repo, "ext", "gpu"),
        joinpath(repo, "gpu"),
        joinpath(repo, "benchmark"),
        joinpath(repo, "docs", "generate_assets", "generate_assets.jl"),
    ]
    files = _script_files_under(paths)

    Test.@test !isfile(joinpath(repo, "ext", "gpu", "policy.jl"))

    stale_tuple_names = Pair{String, Int}[]
    public_tuple_calls = Pair{String, Int}[]
    production_padding_helpers = Pair{String, Int}[]
    unwrapped_device_indices = Pair{String, Int}[]

    for file in files
        for (line_no, line) in enumerate(eachline(file))
            if occursin(r"\b[ux]_tup(?:le)?\b", line)
                push!(stale_tuple_names, file => line_no)
            end
            if occursin("calculate_structure_function", line) &&
                    occursin(r"\b[ux]_tup(?:le)?\b", line)
                push!(public_tuple_calls, file => line_no)
            end
            if startswith(file, joinpath(repo, "ext")) &&
               occursin(r"pad3|_pad3|padded matrix|3D padding", line)
                push!(production_padding_helpers, file => line_no)
            end
            # Two device-index footguns, both invisible to KA.CPU-backend tests:
            #
            # (a) On CUDA, @index(Local/Group, Linear) returns Int32 (threadIdx/blockIdx),
            #     so device helpers must NOT annotate thread/block index params ::Int
            #     (Int64): the call has no matching method, and the GPU compiler turns
            #     that MethodError into an InvalidIRError at kernel compile time
            #     (gpu_gc_pool_alloc / jl_f_throw_methoderror). Use ::Integer and coerce
            #     internally (g = Int(lid)).
            #
            # (b) @index must be bound BARE (`lid = @index(Local, Linear)`): KA's CPU
            #     transform only recognizes that exact assignment form when splicing the
            #     per-workitem argument, so wrapping (`lid = Int(@index(...))`) silently
            #     breaks every kernel on the CPU backend.
            if startswith(file, joinpath(repo, "ext")) &&
               occursin(r"\b(?:lid|bid|block_id|launch_block)::Int\b", line)
                push!(unwrapped_device_indices, file => line_no)
            end
            if startswith(file, joinpath(repo, "ext")) &&
               occursin(r"\w+\(\s*@index\(", line)
                push!(unwrapped_device_indices, file => line_no)
            end
        end
    end

    Test.@test isempty(stale_tuple_names)
    Test.@test isempty(public_tuple_calls)
    Test.@test isempty(production_padding_helpers)
    Test.@test isempty(unwrapped_device_indices)
end
