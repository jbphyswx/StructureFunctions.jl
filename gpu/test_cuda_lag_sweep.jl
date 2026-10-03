# The gridded direct lag sweep on real CUDA: the distance histogram, the joint histograms over angle and
# over value, single slices and batches, into device buffers. This kernel calls the operator per cell pair
# rather than contracting polynomial moments, so it is the only device route to a non-polynomial operator
# or a value histogram on a grid, and only a device compile establishes that the whole reduction builds.
using CUDA: CUDA
using Random: Random
using Printf: Printf
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()

failures = String[]

function compare(name, gs, gc, rs, rc; rtol = 1e-10)
    ds = maximum(abs.(Array(gs) .- rs)) / (maximum(abs, rs) + eps())
    dc = maximum(abs.(float.(Array(gc)) .- float.(rc))) / (maximum(abs, float.(rc)) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    Printf.@printf("%-54s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260918)
const NB = 6
const OPS = (("L2", SFT.L2SFType()), ("S3", SFT.S3SFType()),
             ("FullVector{3}", SFT.FullVectorStructureFunctionType{3}()))

uni = SFC.UniformLagSchedule((16, 16), (1 / 16, 1 / 16), (true, true))
uni_u = reshape(randn(2, 16, 16), 2, :)
uni_b = collect(range(0.0, 0.4; length = NB + 1))

ys = collect(range(0.0, 1.0; length = 11))
rect = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.12,), (false,)), (ys,), (2, 1))
rect_u = reshape(randn(2, 9, 11), 2, :)
rect_b = collect(range(0.0, 0.7; length = NB + 1))

nlat, nlon = 9, 16
lats = collect(range(-70.0, 70.0; length = nlat)) .* (π / 180)
zon = SFC.ZonalLagSchedule(lats, nlon, 2π / nlon, 1.0, true)
zon_u = reshape(randn(2, nlon, nlat), 2, :)
zon_b = collect(range(0.0, 1.5; length = NB + 1))

# The zonal pairs do not share one lag box, so the device offers the box over all pairs and lets
# `_lag_visit` reject; that branch runs only where the trait is false.
Printf.@printf("uniform_lag_box: uniform=%s rectilinear=%s zonal=%s\n",
        SFC.uniform_lag_box(uni), SFC.uniform_lag_box(rect), SFC.uniform_lag_box(zon))

const VAX = SFC.InvariantValueAxis()
const AAX = SFC.SeparationAngleAxis([1.0, 0.0])
const VBINS = collect(range(-2.0, 2.0; length = 9))
const ABINS = collect(range(prevfloat(0.0), π; length = 5))
const NT = 3

for (sname, s, u, edges) in (("uniform", uni, uni_u, uni_b),
                             ("rectilinear", rect, rect_u, rect_b),
                             ("zonal", zon, zon_u, zon_b))
    n = size(u, 2)
    ub = randn(2, n, NT)
    w = 0.5 .+ rand(n)
    axes = s isa SFC.ZonalLagSchedule ? (("value", VAX, VBINS),) : (("value", VAX, VBINS), ("angle", AAX, ABINS))
    for (oname, op) in OPS, (wname, wt, CT) in (("", nothing, Int), (" weighted", w, Float64))
        rs, rc = zeros(NB), zeros(CT, NB)
        SFC.gridded_lag_sweep!(rs, rc, op, u, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = SER)
        gs, gc = CUDA.zeros(Float64, NB), CUDA.zeros(CT, NB)
        SFC.gridded_lag_sweep!(gs, gc, op, u, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = DEV)
        compare("lag sweep $sname $oname$wname", gs, gc, rs, rc)
        rb, rcb = zeros(NB, NT), zeros(CT, NB, NT)
        SFC.gridded_lag_sweep_batch!(rb, rcb, op, ub, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = SER)
        gb, gcb = CUDA.zeros(Float64, NB, NT), CUDA.zeros(CT, NB, NT)
        SFC.gridded_lag_sweep_batch!(gb, gcb, op, ub, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = DEV)
        compare("batch $sname $oname$wname", gb, gcb, rb, rcb)
        for (aname, ax, ab) in axes
            na = length(ab) - 1
            rj, rcj = zeros(NB, na), zeros(NB, na)
            SFC.gridded_lag_sweep!(rj, rcj, op, u, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                   second_axis = ax, backend = SER)
            gj, gcj = CUDA.zeros(Float64, NB, na), CUDA.zeros(Float64, NB, na)
            SFC.gridded_lag_sweep!(gj, gcj, op, u, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                   second_axis = ax, backend = DEV)
            compare("joint $aname $sname $oname$wname", gj, gcj, rj, rcj)
            rj3, rc3 = zeros(NB, na, NT), zeros(NB, na, NT)
            SFC.gridded_lag_sweep_batch!(rj3, rc3, op, ub, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                         second_axis = ax, backend = SER)
            gj3, gc3 = CUDA.zeros(Float64, NB, na, NT), CUDA.zeros(Float64, NB, na, NT)
            SFC.gridded_lag_sweep_batch!(gj3, gc3, op, ub, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                         second_axis = ax, backend = DEV)
            compare("batch joint $aname $sname $oname$wname", gj3, gc3, rj3, rc3)
        end
    end
end

if isempty(failures)
    println("\nCUDA LAG SWEEP OK")
else
    println("\nCUDA LAG SWEEP FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
