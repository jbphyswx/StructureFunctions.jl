# Executable examples

Each example uses 32 points and a fixed random seed. The random fields demonstrate
the API; they do not model an inertial range. Set `SF_EXAMPLE_POINTS` explicitly
for larger inputs. Pair calculations can require quadratic work.

| File | Purpose | Environment |
|---|---|---|
| `simple_2d.jl` | Second-order vector SF and bin separations | `examples` |
| `single_pass.jl` | Six moments sharing pair geometry | `examples` |
| `threaded_calculation.jl` | Serial/threaded numerical agreement | `examples`, multiple threads |
| `distributed_parallel.jl` | Local workers, reduction, and owned-worker cleanup | `examples`, at least three allocated CPUs |
| `gpu_acceleration.jl` | Reuse GPU histogram scratch | `gpu`, allocated CUDA device |
| `gpu_time_slices.jl` | Batched device inputs and per-snapshot reference checks | `gpu`, allocated CUDA device |

From the repository root, start Julia with `julia --project=examples`. Install the
environment once with `using Pkg; Pkg.instantiate()`, then run examples in that
session with `include("examples/simple_2d.jl")`. GPU examples use `--project=gpu`.
They fail if CUDA is unavailable. Each script also defines a function for repeated
execution after its initial demonstration.

The threaded and Distributed examples need several CPUs; start Julia with
`--threads=N` for the threaded example. The GPU examples need a CUDA device; on a
cluster, request one through the scheduler before starting Julia.

The Distributed example starts only the missing local workers with the active
project and one thread each. It preserves existing workers and removes the ones
it created in a `finally` block. Existing workers must use the same project.

See the [data guide](../docs/src/data.md), [backend guide](../docs/src/backends.md),
and [mathematical definitions](../docs/src/theory.md) for shapes and conventions.
