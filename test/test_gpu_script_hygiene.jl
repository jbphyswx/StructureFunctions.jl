using Test: Test

"""The `file => line` of every line of the Julia sources under `root` that matches `pattern`."""
function _lines_matching(root, pattern)
    files = sort!([joinpath(dir, name) for (dir, _, names) in walkdir(root) for name in names if endswith(name, ".jl")])
    return [file => n for file in files for (n, line) in enumerate(eachline(file)) if occursin(pattern, line)]
end

# Two device-index pitfalls anywhere in the extension sources, including kernels the CPU-backend suite never launches.
Test.@testset "GPU script hygiene" begin
    ext = normpath(joinpath(@__DIR__, "..", "ext"))
    # @index is Int32 on CUDA, so a device index parameter typed ::Int has no method there.
    Test.@test isempty(_lines_matching(ext, r"\b(?:lid|bid|block_id|launch_block)::Int\b"))
    # KA's CPU backend supplies the work-item index only to a bare `x = @index(...)` binding.
    Test.@test isempty(_lines_matching(ext, r"\w+\(\s*@index\("))
end
