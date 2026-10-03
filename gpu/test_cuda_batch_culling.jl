# Culling on the device slice batch on real CUDA: shared positions sorted once into the workspace's cull
# memo for every slice, positions varying per slice culled one slice at a time, on the native kernels
# (width 2) and the portable ones (width 7), and the moment tensor and multi-field through the same
# memo. Every policy gives the serial answer.
using CUDA: CUDA
using Random: Random
using Printf: Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.MultiFields: MultiFields as MF
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const BE = CUDA.CUDABackend()
const DEV = CB.GPUBackend(BE)
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()

failures = String[]

function check(name, got_s, got_c, ref_s, ref_c, engaged)
    gs, gc = Array(got_s), Array(got_c)
    counts_ok = eltype(ref_c) <: Integer ? gc == ref_c : isapprox(gc, ref_c; rtol = 1e-10)
    ok = counts_ok && isapprox(gs, ref_s; rtol = 1e-9, atol = 1e-12) && engaged && sum(ref_c) > 0
    ok || push!(failures, name)
    Printf.@printf("%-72s %s\n", name, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260926)
const N, B = 3000, 4
const VAL = collect(range(-4.0, 4.0; length = 9))
const VB = ntuple(_ -> VAL, SFC.SINGLE_PASS_N)
const NV, NI = length(VAL) - 1, SFC.SINGLE_PASS_N

for (D, hi) in ((2, 0.06), (7, 0.3)), shared in (true, false), weighted in (false, true)
    bins = collect(range(0.0, hi; length = 9))
    nb = length(bins) - 1
    x = shared ? rand(D, N) : rand(D, N, B)
    u = randn(D, N, B)
    kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
    CT = weighted ? Float64 : UInt32
    families = (
        (:batch1d, (nb, B), () -> SFC.GPUSFWorkspace(BE, bins),
         (s, c, be, ws, pol) -> SFC.calculate_structure_function_batch!(
             s, c, OP, x, u, bins; backend = be, workspace = ws, culling = pol, kw...)),
        (:joint, (nb, NV, B), () -> SFC.GPUSFWorkspace(BE, bins, VAL; kind = :joint2d),
         (s, c, be, ws, pol) -> SFC.calculate_structure_function_2d_batch!(
             s, c, OP, x, u, bins, VAL; backend = be, workspace = ws, culling = pol, kw...)),
        (:sp1d, (NI, nb, B), () -> SFC.GPUSFWorkspace(BE, bins; kind = :single_pass),
         (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_batch!(
             s, c, x, u, bins; backend = be, workspace = ws, culling = pol, kw...)),
        (:sp2d, (NI, nb, NV, B), () -> SFC.GPUSFWorkspace(BE, bins, VB; kind = :single_pass_2d),
         (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_2d_batch!(
             s, c, x, u, bins, VB; backend = be, workspace = ws, culling = pol, kw...)),
    )
    for (name, shape, workspace, run!) in families
        rs, rc = zeros(shape), zeros(CT, shape)
        run!(rs, rc, SER, nothing, SFC.NoCulling())
        for pol in (SFC.AutoCulling(), SFC.AlwaysCulling())
            ws = workspace()
            gs, gc = CUDA.zeros(Float64, shape...), CUDA.zeros(CT, shape...)
            run!(gs, gc, DEV, ws, pol)
            check("$name D=$D shared=$shared weighted=$weighted $(nameof(typeof(pol)))",
                  gs, gc, rs, rc, ws.lazy.cull isa SFC.GPUCullMemo)
        end
    end
end

# The moment tensor and the multi-field run the same tiled kernels through the same workspace memo.
let D = 2, bins = collect(range(0.0, 0.06; length = 9))
    nb = length(bins) - 1
    for weighted in (false, true)
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        for (layout, x) in (("shared", rand(D, N)), ("varying", rand(D, N, B))), P in (2, 3, 4)
            u = randn(D, N, B)
            rs, rc = zeros(ntuple(_ -> D, P)..., nb, B), zeros(CT, nb, B)
            SFC.calculate_structure_function_tensor!(rs, rc, Val(P), x, u, bins; backend = SER, kw...)
            ws = SFC.GPUSFWorkspace(BE, bins)
            gs, gc = CUDA.zeros(Float64, size(rs)...), CUDA.zeros(CT, size(rc)...)
            SFC.calculate_structure_function_tensor!(gs, gc, Val(P), x, u, bins; backend = DEV, workspace = ws,
                                                     culling = SFC.AlwaysCulling(), kw...)
            check("tensor P=$P $layout weighted=$weighted", gs, gc, rs, rc, ws.lazy.cull isa SFC.GPUCullMemo)
        end
        x = rand(D, N)
        f = MF.Fields(vectors = (randn(D, N), randn(D, N)), scalars = (randn(N),))
        for op in (SFT.MixedSFType{1, 0, 2}(), SFT.VectorDotSFType(1, 2))
            rs, rc = zeros(nb), zeros(CT, nb)
            SFC.calculate_structure_function!(rs, rc, op, x, f, bins; backend = SER, kw...)
            ws = SFC.GPUSFWorkspace(BE, bins)
            gs, gc = CUDA.zeros(Float64, nb), CUDA.zeros(CT, nb)
            SFC.calculate_structure_function!(gs, gc, op, x, f, bins; backend = DEV, workspace = ws,
                                              culling = SFC.AlwaysCulling(), kw...)
            check("multi-field $(nameof(typeof(op))) weighted=$weighted", gs, gc, rs, rc,
                  ws.lazy.cull isa SFC.GPUCullMemo)
        end
    end
end

if isempty(failures)
    println("\nCUDA BATCH CULLING OK")
else
    println("\nCUDA BATCH CULLING FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
