using StructureFunctions
using Aqua
using Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        StructureFunctions;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        project_extras = true,
        stale_deps = true,
        deps_compat = true,
    )
end
