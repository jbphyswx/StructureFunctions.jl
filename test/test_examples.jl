using Test: Test

Test.@testset "Small serial examples" begin
    for name in ("simple_2d.jl", "single_pass.jl")
        sandbox = Core.eval(Main, Expr(:module, true, gensym(:Example), Expr(:block)))
        result = Base.invokelatest(Base.include, sandbox, joinpath(@__DIR__, "..", "examples", name))
        Test.@test result !== nothing
    end
end
