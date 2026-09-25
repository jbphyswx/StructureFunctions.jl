using Test: Test

# Execute the small README examples in order, sharing their documented bindings.

const README_PATH = joinpath(@__DIR__, "..", "README.md")

"""Every ` ```julia ` block of `path`, in order, with the installation line removed."""
function readme_julia_blocks(path::AbstractString)
    blocks = String[]
    inside = false
    buf = IOBuffer()
    for line in eachline(path)
        if !inside && startswith(line, "```julia")
            inside = true
            truncate(buf, 0)
        elseif inside && startswith(line, "```")
            inside = false
            push!(blocks, String(take!(buf)))
        elseif inside
            occursin("Pkg.add", line) || println(buf, line)
        end
    end
    return blocks
end

Test.@testset "every README code block runs" begin
    blocks = readme_julia_blocks(README_PATH)
    Test.@test length(blocks) >= 3

    sandbox = Module(:READMESandbox)
    Core.eval(sandbox, :(eval(x) = Core.eval($sandbox, x)))
    for (i, block) in enumerate(blocks)
        Test.@test (i, begin
            Base.include_string(sandbox, block, "README.md block $i")
            true
        end) == (i, true)
    end
end
