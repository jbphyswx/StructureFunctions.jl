# MPI backend multi-rank parity: `mpiexec -n 2` on mpi_rankscript.jl, asserting its exit code.
# The child's stdout/stderr are this process's.

using Test: Test
using MPI: MPI

Test.@testset "MPI backend (multi-rank parity)" begin
    script = joinpath(@__DIR__, "mpi_rankscript.jl")
    proj = Base.active_project()
    # `mpiexec` establishes the launcher environment for the duration of the block.
    parity_ok = MPI.mpiexec() do exe
        cmd = `$exe -n 2 $(Base.julia_cmd()) --project=$proj --threads=2 $script`
        success(pipeline(cmd; stdout = stdout, stderr = stderr))
    end
    Test.@test parity_ok
end