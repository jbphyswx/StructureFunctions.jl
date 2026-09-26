# The device batch contract on CUDA: every primary family over no, one and two trailing batch axes,
# shared and varying positions, weighted and unweighted, with and without a workspace. Results stay on
# the device and equal serial; `!` twice is twice the histogram, into count buffers the kernels
# accumulate in directly (`UInt32` unweighted, the weighted count type) and into ones they do not.
using Test: Test
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

CUDA.functional() || error("the device batch contract needs a functional CUDA device")
CUDA.allowscalar(false)

Test.@testset "device batch contract" begin
    rng = Random.MersenneTwister(70)
    bins = range(0f0, 1.5f0; length = 5)
    vb = range(-40f0, 40f0; length = 5)
    device = CUDA.CUDABackend()
    backend = CB.GPUBackend(device)
    op = SFT.S2SFType()
    RAW = SF.StructureFunctionSumsAndCounts
    for dims in ((), (2,), (2, 2)), fixed in (false, true), weighted in (false, true), prepared in (false, true)
        isempty(dims) && fixed && continue
        u = randn(rng, Float32, 2, 16, dims...)
        x = fixed ? rand(rng, Float32, 2, 16) : rand(rng, Float32, size(u)...)
        dx, du = CUDA.CuArray(x), CUDA.CuArray(u)
        w = weighted ? rand(rng, Float32, 16) : nothing
        dw = weighted ? CUDA.CuArray(w) : nothing
        for CT in (weighted ? (Float64,) : (UInt32, UInt64)), family in (:individual, :joint, :single, :single_joint)
            ws = !prepared ? nothing :
                 family === :individual ? SFC.GPUSFWorkspace(device, bins) :
                 family === :joint ? SFC.GPUSFWorkspace(device, bins, vb) :
                 family === :single ? SFC.GPUSFWorkspace(device, bins; kind = :single_pass) :
                 SFC.GPUSFWorkspace(device, bins, vb; kind = :single_pass_2d)
            wkw(workspace) = workspace === nothing ? (;) : (; workspace)
            calc(a, b, be, weights, workspace) =
                family === :individual ?
                    SFC.calculate_structure_function(op, a, b, bins, CT, RAW; backend = be, weights, wkw(workspace)...) :
                family === :joint ?
                    SFC.calculate_structure_function(op, a, b, bins, vb, CT; backend = be, weights, wkw(workspace)...) :
                family === :single ?
                    SFC.calculate_structure_functions_single_pass(a, b, bins, CT; backend = be, weights,
                                                                   wkw(workspace)...) :
                    SFC.calculate_structure_functions_single_pass_2d(a, b, bins, vb, CT; backend = be, weights,
                                                                      wkw(workspace)...)
            result = calc(dx, du, backend, dw, ws)
            reference = calc(x, u, CB.SerialBackend(), w, nothing)
            names = family in (:individual, :joint) ? (:only,) : keys(SFC.SINGLE_PASS_OPERATORS)
            case = (dims, fixed, weighted, prepared, CT, family)
            for name in names
                r = name === :only ? result : result[name]
                ref = name === :only ? reference : reference[name]
                Test.@test (case, KA.get_backend(r.sums) isa CUDA.CUDABackend) == (case, true)
                host = SF.to_host(r)
                Test.@test (case, isapprox(host.sums, ref.sums; rtol = 5f-5, atol = 1f-5)) == (case, true)
                Test.@test (case, isapprox(host.counts, ref.counts; rtol = 5f-5)) == (case, true)
            end
            shape = family === :individual ? (4, dims...) : family === :joint ? (4, 4, dims...) :
                    family === :single ? (6, 4, dims...) : (6, 4, 4, dims...)
            sums, counts = CUDA.zeros(Float32, shape...), CUDA.zeros(CT, shape...)
            for _ in 1:2
                kw = (; backend, weights = dw, wkw(ws)...)
                if family === :individual
                    SFC.calculate_structure_function!(sums, counts, op, dx, du, bins; kw...)
                elseif family === :joint
                    SFC.calculate_structure_function!(sums, counts, op, dx, du, bins, vb; kw...)
                elseif family === :single
                    SFC.calculate_structure_functions_single_pass!(sums, counts, dx, du, bins; kw...)
                else
                    SFC.calculate_structure_functions_single_pass_2d!(sums, counts, dx, du, bins, vb; kw...)
                end
            end
            hs, hc = Array(sums), Array(counts)
            for (i, name) in enumerate(names)
                ref = name === :only ? reference : reference[name]
                rs = name === :only ? hs : selectdim(hs, 1, i)
                rc = name === :only ? hc : selectdim(hc, 1, i)
                Test.@test (case, name, isapprox(rs, 2 .* ref.sums; rtol = 5f-5, atol = 2f-5)) == (case, name, true)
                Test.@test (case, name, isapprox(rc, 2 .* ref.counts; rtol = 5f-5)) == (case, name, true)
                # An allocated result owns its buffers: accumulating afterwards leaves it unchanged.
                preserved = SF.to_host(name === :only ? result : result[name])
                Test.@test (case, name, isapprox(preserved.sums, ref.sums; rtol = 5f-5, atol = 1f-5)) ==
                           (case, name, true)
            end
            ws === nothing || SFC.release!(ws)
        end
    end
end
