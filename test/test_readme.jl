using Test: Test

# The README is not part of the Documenter build, so nothing executes what it shows. Two of its
# samples were broken for as long as anyone had gone without reading them: one named a submodule
# path that raises `UndefVarError`, the other called an unqualified entry that does not resolve.
# Both are the kind of defect that only running the code finds, so this file runs it.
#
# The blocks are run verbatim and in order, sharing one module, which is what a reader copying the
# page top to bottom does — so a later block reading a name an earlier one bound is covered too.
# The only line dropped is the installation command, which would install the package over itself.

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
