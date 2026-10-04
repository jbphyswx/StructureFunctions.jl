using Test: Test
using Random: Random
using StructureFunctions: Calculations as SFC, LinearBinEdges
using StructureFunctions.Calculations: CPUSFWorkspace
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads

# A CPUSFWorkspace reused across calls leaves single-pass batch results unchanged and rejects workspaces that do not fit the call.
Test.@testset "CPUSFWorkspace" begin
    Random.seed!(4242)
    N, B, nd, nv = 60, 3, 8, 6
    x = rand(2, N)
    u = rand(2, N, B)
    db = LinearBinEdges(range(0.0, 2.0; length = nd + 1))
    vb = LinearBinEdges(range(-3.0, 3.0; length = nv + 1))
    backends = CB.ThreadedBackend[]
    Threads.nthreads() > 1 && push!(backends, CB.ThreadedBackend())

    same_sums(a, b, backend) = backend isa CB.SerialBackend ? a == b : isapprox(a, b; rtol = 1e-12)
    # Two calls through one workspace each equal the call without one: exactly on serial, to rounding threaded.
    Test.@testset "reuse across calls leaves results unchanged" begin
        for backend in (CB.SerialBackend(), backends...)
            ws = CPUSFWorkspace{:single_pass_2d}(x, u, db, vb; backend)
            ref = (zeros(Float64, 6, nd, nv, B), zeros(UInt32, 6, nd, nv, B))
            SFC.calculate_structure_functions_single_pass_2d_batch!(ref..., x, u, db, vb; backend)
            runs = map(1:2) do _
                out = (zeros(Float64, 6, nd, nv, B), zeros(UInt32, 6, nd, nv, B))
                SFC.calculate_structure_functions_single_pass_2d_batch!(out..., x, u, db, vb; backend, workspace = ws)
                out
            end
            Test.@test all(r -> same_sums(r[1], ref[1], backend) && r[2] == ref[2], runs)

            ws1 = CPUSFWorkspace{:single_pass}(x, u, db; backend)
            ref1 = (zeros(Float64, 6, nd, B), zeros(UInt32, 6, nd, B))
            SFC.calculate_structure_functions_single_pass_batch!(ref1..., x, u, db; backend)
            runs1 = map(1:2) do _
                out = (zeros(Float64, 6, nd, B), zeros(UInt32, 6, nd, B))
                SFC.calculate_structure_functions_single_pass_batch!(out..., x, u, db; backend, workspace = ws1)
                out
            end
            Test.@test all(r -> same_sums(r[1], ref1[1], backend) && r[2] == ref1[2], runs1)
        end
    end

    # A workspace built from a BatchLeading array reproduces the BatchLeading result without one.
    Test.@testset "composes with BatchLeading" begin
        ubl = SFC.BatchLeading(permutedims(u, (3, 1, 2)))
        ws = CPUSFWorkspace{:single_pass_2d}(x, ubl, db, vb)
        s1 = zeros(Float64, 6, nd, nv, B); c1 = zeros(UInt32, 6, nd, nv, B)
        s2 = zeros(Float64, 6, nd, nv, B); c2 = zeros(UInt32, 6, nd, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            s1, c1, x, ubl, db, vb; backend = CB.SerialBackend())
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            s2, c2, x, ubl, db, vb; backend = CB.SerialBackend(), workspace = ws)
        Test.@test s1 == s2
        Test.@test c1 == c2
    end

    # A workspace with the wrong N, B, bin count or kind, or an unknown kind, throws ArgumentError.
    Test.@testset "a mismatched workspace is a hard error" begin
        s = zeros(Float64, 6, nd, nv, B); c = zeros(UInt32, 6, nd, nv, B)
        call!(ws) = SFC.calculate_structure_functions_single_pass_2d_batch!(
            s, c, x, u, db, vb; backend = CB.SerialBackend(), workspace = ws)

        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            rand(2, N + 5), rand(2, N + 5, B), db, vb))
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            x, rand(2, N, B + 1), db, vb))
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            x, u, LinearBinEdges(range(0.0, 2.0; length = nd + 3)), vb))
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass}(x, u, db))
        Test.@test_throws ArgumentError CPUSFWorkspace{:nonsense}(x, u, db, vb)
    end
end
