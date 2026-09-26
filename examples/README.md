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

On clima, GPU use and parallel CPU work require a Slurm allocation. For an
interactive threaded session:

```bash
salloc --cpus-per-task=4 --mem=4G --time=00:30:00
srun --pty env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 julia --project=examples --threads=4
```

Request `--gres=gpu:A100:1` for CUDA work and use the GPU project. The example
resource guard verifies the running owned allocation, node, step, CPU count,
and assigned GPU indices before importing CUDA.

Codex development on clima uses the installed persistent Julia skill:

```bash
python3 ~/.codex/skills/julia-repl/scripts/jlrepl.py start --owner examples-cpu \
  --project examples --mode heavy --slurm --threads 4 --memory 4G --time 00:30:00
python3 ~/.codex/skills/julia-repl/scripts/jlrepl.py run --owner examples-cpu \
  --project examples examples/distributed_parallel.jl
python3 ~/.codex/skills/julia-repl/scripts/jlrepl.py start --owner examples-gpu \
  --project gpu --mode gpu --slurm --threads 2 --memory 8G --time 00:30:00
python3 ~/.codex/skills/julia-repl/scripts/jlrepl.py run --owner examples-gpu \
  --project gpu examples/gpu_acceleration.jl
```

The Distributed example starts only the missing local workers with the active
project and one thread each. It preserves existing workers and removes the ones
it created in a `finally` block. Existing workers must use the same project.

See the [data guide](../docs/src/data.md), [backend guide](../docs/src/backends.md),
and [mathematical definitions](../docs/src/theory.md) for shapes and conventions.
