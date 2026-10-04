using Test: Test

Test.@testset "Small serial examples" begin
    for name in ("simple_2d.jl", "single_pass.jl")
        sandbox = Core.eval(Main, Expr(:module, true, gensym(:Example), Expr(:block)))
        failure = try
            Base.invokelatest(Base.include, sandbox, joinpath(@__DIR__, "..", "examples", name))
            nothing
        catch e
            (name, e)
        end
        Test.@test failure === nothing
    end
end
