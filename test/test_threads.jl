using Test: Test

Test.@testset "Threading Backends" begin
    test_project = abspath(@__DIR__)
    script = joinpath(test_project, "threads_subprocess_smoke.jl")

    cmd_1 = `$(Base.julia_cmd()) --threads=1 --startup-file=no --project=$(test_project) $(script)`
    Test.@test success(pipeline(cmd_1, stderr = stderr, stdout = stdout))

    cmd_2 = `$(Base.julia_cmd()) --threads=2 --startup-file=no --project=$(test_project) $(script)`
    Test.@test success(pipeline(cmd_2, stderr = stderr, stdout = stdout))
end
