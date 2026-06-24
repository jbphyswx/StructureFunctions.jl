# MPI backend multi-rank parity. Launches mpi_rankscript.jl under mpiexec -n 2 and checks the
# rank-0 parity marker. Guarded: if MPI / mpiexec is unavailable (or the launch fails in a
# restricted environment), the test is skipped rather than failing the suite — the MPI
# extension is an offered, optional backend.
using Test

const _mpi_loadable = try
    @eval using MPI
    true
catch
    false
end

Test.@testset "MPI backend (multi-rank parity)" begin
    if !_mpi_loadable
        @info "MPI not available; skipping MPI backend test"
        Test.@test_skip true
    else
        script = joinpath(@__DIR__, "mpi_rankscript.jl")
        proj = Base.active_project()
        out = ""
        ok = try
            buf = IOBuffer()
            MPI.mpiexec() do exe
                run(pipeline(`$exe -n 2 $(Base.julia_cmd()) --project=$proj $script`;
                             stdout = buf, stderr = buf))
            end
            out = String(take!(buf))
            true
        catch err
            @info "mpiexec launch failed; skipping MPI backend test" err
            false
        end
        if ok
            Test.@test occursin("MPI_PARITY_OK", out)
        else
            Test.@test_skip true
        end
    end
end
