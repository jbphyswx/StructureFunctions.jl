using StructureFunctions: StructureFunctions as SF
using Aqua: Aqua
using Test: Test

Test.@testset "Aqua.jl" begin
    Aqua.test_all(
        SF;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        project_extras = true,
        stale_deps = true,
        deps_compat = true,
    )
end
