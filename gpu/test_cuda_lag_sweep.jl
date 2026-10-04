using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const NB = 6
const NT = 3
const VAX = SFC.InvariantValueAxis()
const AAX = SFC.SeparationAngleAxis([1.0, 0.0])
const VBINS = collect(range(-2.0, 2.0; length = 9))
const ABINS = collect(range(prevfloat(0.0), π; length = 5))
const OPS = Dict("L2" => SFT.L2SFType(), "S3" => SFT.S3SFType(),
                 "FullVector{3}" => SFT.FullVectorStructureFunctionType{3}())

function compare(gs, gc, rs, rc; rtol = 1e-10)
    Test.@test maximum(abs.(Array(gs) .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(float.(Array(gc)) .- float.(rc))) / (maximum(abs, float.(rc)) + eps()) < rtol
end

"""The schedule, a one-slice field, and the distance edges of the grid `name`."""
function grid(name)
    if name == "uniform"
        return (SFC.UniformLagSchedule((16, 16), (1 / 16, 1 / 16), (true, true)), reshape(randn(2, 16, 16), 2, :),
                collect(range(0.0, 0.4; length = NB + 1)))
    elseif name == "rectilinear"
        ys = collect(range(0.0, 1.0; length = 11))
        return (SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.12,), (false,)), (ys,), (2, 1)),
                reshape(randn(2, 9, 11), 2, :), collect(range(0.0, 0.7; length = NB + 1)))
    elseif name == "zonal"
        nlat, nlon = 9, 16
        lats = collect(range(-70.0, 70.0; length = nlat)) .* (π / 180)
        return (SFC.ZonalLagSchedule(lats, nlon, 2π / nlon, 1.0, true), reshape(randn(2, nlon, nlat), 2, :),
                collect(range(0.0, 1.5; length = NB + 1)))
    end
    error("not implemented: grid $name")
end

# (grid, operator, weighted): every operator on a flat grid and on the sphere, each weight with the angle axis
const CASES = (
    ("uniform", "L2", false),
    ("uniform", "FullVector{3}", true),
    ("rectilinear", "S3", true),
    ("zonal", "L2", true),
    ("zonal", "S3", false),
    ("zonal", "FullVector{3}", false),
)

Test.@testset "gridded direct lag sweep on the device" begin
    Random.seed!(20260918)

    # The zonal pairs do not share one lag box, so only the zonal grid takes the branch that rejects a lag per pair.
    Test.@testset "uniform lag box" begin
        Test.@test SFC.uniform_lag_box(first(grid("uniform")))
        Test.@test SFC.uniform_lag_box(first(grid("rectilinear")))
        Test.@test !SFC.uniform_lag_box(first(grid("zonal")))
    end

    # The distance histogram and the joint histograms over value and angle, one slice and a batch, into device buffers.
    Test.@testset "$sname $oname weighted=$weighted" for (sname, oname, weighted) in CASES
        s, u, edges = grid(sname)
        op = OPS[oname]
        n = size(u, 2)
        ub = randn(2, n, NT)
        wt, CT = weighted ? (0.5 .+ rand(n), Float64) : (nothing, Int)
        axes = s isa SFC.ZonalLagSchedule ? (("value", VAX, VBINS),) : (("value", VAX, VBINS), ("angle", AAX, ABINS))
        Test.@testset "distance" begin
            rs, rc = zeros(NB), zeros(CT, NB)
            SFC.gridded_lag_sweep!(rs, rc, op, u, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = SER)
            gs, gc = CUDA.zeros(Float64, NB), CUDA.zeros(CT, NB)
            SFC.gridded_lag_sweep!(gs, gc, op, u, s, edges, Val(2), Val(1), Val(0); weights = wt, backend = DEV)
            compare(gs, gc, rs, rc)
        end
        Test.@testset "distance batch" begin
            rb, rcb = zeros(NB, NT), zeros(CT, NB, NT)
            SFC.gridded_lag_sweep_batch!(rb, rcb, op, ub, s, edges, Val(2), Val(1), Val(0); weights = wt,
                                         backend = SER)
            gb, gcb = CUDA.zeros(Float64, NB, NT), CUDA.zeros(CT, NB, NT)
            SFC.gridded_lag_sweep_batch!(gb, gcb, op, ub, s, edges, Val(2), Val(1), Val(0); weights = wt,
                                         backend = DEV)
            compare(gb, gcb, rb, rcb)
        end
        Test.@testset "joint $aname" for (aname, ax, ab) in axes
            na = length(ab) - 1
            rj, rcj = zeros(NB, na), zeros(NB, na)
            SFC.gridded_lag_sweep!(rj, rcj, op, u, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                   second_axis = ax, backend = SER)
            gj, gcj = CUDA.zeros(Float64, NB, na), CUDA.zeros(Float64, NB, na)
            SFC.gridded_lag_sweep!(gj, gcj, op, u, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                   second_axis = ax, backend = DEV)
            Test.@testset "one slice" begin
                compare(gj, gcj, rj, rcj)
            end
            rj3, rc3 = zeros(NB, na, NT), zeros(NB, na, NT)
            SFC.gridded_lag_sweep_batch!(rj3, rc3, op, ub, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                         second_axis = ax, backend = SER)
            gj3, gc3 = CUDA.zeros(Float64, NB, na, NT), CUDA.zeros(Float64, NB, na, NT)
            SFC.gridded_lag_sweep_batch!(gj3, gc3, op, ub, s, edges, ab, Val(2), Val(1), Val(0); weights = wt,
                                         second_axis = ax, backend = DEV)
            Test.@testset "batch" begin
                compare(gj3, gc3, rj3, rc3)
            end
        end
    end
end
