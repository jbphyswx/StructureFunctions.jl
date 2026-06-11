# StructureFunctions.jl Profiling Suite

This directory contains scripts and tools for profiling CPU execution time and memory allocation patterns in `StructureFunctions.jl`.

## How to Run

To run the standard profiling suite (using 24 threads, $N = 30,000$ points, and 50 log-spaced bins):

```bash
julia -t 24 --project=benchmark benchmark/profile/profile_runner.jl
```

## Generated Outputs

Running the profiling suite outputs the following files in this directory:

### CPU Profiles
* `cpu_serial.txt` & `cpu_threaded.txt`: Flat summaries of execution times grouped by source location/functions.
* `cpu_serial.jls` & `cpu_threaded.jls`: Serialized binary files containing the raw stack trace frames from Julia's `Profile` stdlib.

### Memory Allocation Profiles
* `allocs_serial.txt` & `allocs_threaded.txt`: Summaries showing total allocations, top allocated types, and the source code locations causing allocations.
* `allocs_serial.jls` & `allocs_threaded.jls`: Serialized binary files containing raw allocation trace results from `Profile.Allocs`.

---

## How to Visualize Raw Profiling Data

You can load the serialized `.jls` binary files into any active Julia session for rich visual analysis.

### 1. View interactive Flame Graphs (`ProfileView.jl` or `ProfileCanvas.jl`)

In a Julia REPL:
```julia
using Serialization, Profile
using ProfileView # or `using ProfileCanvas`

# Load Serial CPU profile
data = deserialize("benchmark/profile/cpu_serial.jls")

# Restore profile state and view interactively
Profile.init(data...)
ProfileView.view()
```

### 2. Export to `pprof` for web browser visualization (`PProf.jl`)

To view allocations or CPU traces in your web browser:
```julia
using Serialization, PProf

# Load Serial Allocations profile
allocs = deserialize("benchmark/profile/allocs_serial.jls")

# Start local pprof server and open in browser
pprof(allocs)
```
