using Test: Test
using Random: Random
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges
using StructureFunctions.Calculations: CPUSFWorkspace, reset_histogram!
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads

Test.@testset "CPUSFWorkspace" begin
    Random.seed!(4242)
    N, B, nd, nv = 60, 8, 12, 10
    nt = Threads.nthreads()
    x = rand(2, N)
    u = rand(2, N, B)
    db = LinearBinEdges(range(0.0, 2.0; length = nd + 1))
    vb = LinearBinEdges(range(-3.0, 3.0; length = nv + 1))
    sft = SFT.L2SFType()
    backends = CB.ThreadedBackend[]
    Threads.nthreads() > 1 && push!(backends, CB.ThreadedBackend())

    Test.@testset "results identical with and without a workspace" begin
        for backend in (CB.SerialBackend(), backends...)
            ws = CPUSFWorkspace{:single_pass_2d}(x, u, db, vb; backend)
            s1 = zeros(Float64, 6, nd, nv, B); c1 = zeros(UInt32, 6, nd, nv, B)
            s2 = zeros(Float64, 6, nd, nv, B); c2 = zeros(UInt32, 6, nd, nv, B)
            SFC.calculate_structure_functions_single_pass_2d_batch!(s1, c1, x, u, db, vb; backend)
            SFC.calculate_structure_functions_single_pass_2d_batch!(s2, c2, x, u, db, vb; backend, workspace = ws)
            Test.@test s1 == s2
            Test.@test c1 == c2

            ws1 = CPUSFWorkspace{:single_pass}(x, u, db; backend)
            p1 = zeros(Float64, 6, nd, B); q1 = zeros(UInt32, 6, nd, B)
            p2 = zeros(Float64, 6, nd, B); q2 = zeros(UInt32, 6, nd, B)
            SFC.calculate_structure_functions_single_pass_batch!(p1, q1, x, u, db; backend)
            SFC.calculate_structure_functions_single_pass_batch!(p2, q2, x, u, db; backend, workspace = ws1)
            Test.@test p1 == p2
            Test.@test q1 == q2
        end
    end

    Test.@testset "reuse across calls is stable" begin
        ws = CPUSFWorkspace{:single_pass_2d}(x, u, db, vb)
        ref_s = zeros(Float64, 6, nd, nv, B); ref_c = zeros(UInt32, 6, nd, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            ref_s, ref_c, x, u, db, vb; backend = CB.SerialBackend())
        for _ in 1:3
            s = zeros(Float64, 6, nd, nv, B); c = zeros(UInt32, 6, nd, nv, B)
            SFC.calculate_structure_functions_single_pass_2d_batch!(
                s, c, x, u, db, vb; backend = CB.SerialBackend(), workspace = ws)
            Test.@test s == ref_s
            Test.@test c == ref_c
        end
    end

    Test.@testset "composes with BatchLeading (no transpose buffer)" begin
        ubl = SFC.BatchLeading(permutedims(u, (3, 1, 2)))
        ws = CPUSFWorkspace{:single_pass_2d}(x, ubl, db, vb)
        Test.@test length(ws.ub) == 0
        Test.@test length(ws.xb) == 0
        s1 = zeros(Float64, 6, nd, nv, B); c1 = zeros(UInt32, 6, nd, nv, B)
        s2 = zeros(Float64, 6, nd, nv, B); c2 = zeros(UInt32, 6, nd, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            s1, c1, x, ubl, db, vb; backend = CB.SerialBackend())
        SFC.calculate_structure_functions_single_pass_2d_batch!(
            s2, c2, x, ubl, db, vb; backend = CB.SerialBackend(), workspace = ws)
        Test.@test s1 == s2
        Test.@test c1 == c2
    end

    Test.@testset "a mismatched workspace is a hard error" begin
        s = zeros(Float64, 6, nd, nv, B); c = zeros(UInt32, 6, nd, nv, B)
        call!(ws) = SFC.calculate_structure_functions_single_pass_2d_batch!(
            s, c, x, u, db, vb; backend = CB.SerialBackend(), workspace = ws)

        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            rand(2, N + 5), rand(2, N + 5, B), db, vb))          # wrong N
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            x, rand(2, N, B + 1), db, vb))                        # wrong B
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass_2d}(
            x, u, LinearBinEdges(range(0.0, 2.0; length = nd + 3)), vb))  # wrong n_bins
        Test.@test_throws ArgumentError call!(CPUSFWorkspace{:single_pass}(x, u, db))  # wrong kind
        Test.@test_throws ArgumentError CPUSFWorkspace{:nonsense}(x, u, db, vb)
    end

    Test.@testset "concretely typed" begin
        ws = CPUSFWorkspace{:single_pass_2d}(x, u, db, vb)
        T = typeof(ws)
        Test.@test isconcretetype(T)
        Test.@test all(isconcretetype, fieldtypes(T))
        Test.@test reset_histogram!(ws) === ws
        Test.@test all(iszero, ws.result[1])
    end
end

