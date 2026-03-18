using Test: Test
using OhMyThreads: OhMyThreads
using StructureFunctions: StructureFunctions as SF

x = (collect(0.0:1.0:7.0), collect(0.0:1.0:7.0))
u = (collect(1.0:1.0:8.0), collect(2.0:1.0:9.0))
bins = [(0.0, 20.0)]

serial = SF.calculate_structure_function(
    SF.L2SF(),
    x,
    u,
    bins;
    backend = SF.SerialBackend(),
    verbose = false,
    show_progress = false,
)

threaded = SF.calculate_structure_function(
    SF.L2SF(),
    x,
    u,
    bins;
    backend = SF.ThreadedBackend(),
    verbose = false,
    show_progress = false,
)

auto_backend = SF.calculate_structure_function(
    SF.L2SF(),
    x,
    u,
    bins;
    backend = SF.AutoThreadingBackend(),
    verbose = false,
    show_progress = false,
)

default_backend = SF.calculate_structure_function(
    SF.L2SF(),
    x,
    u,
    bins;
    verbose = false,
    show_progress = false,
)

Test.@test serial.values == threaded.values
Test.@test auto_backend.values == threaded.values || auto_backend.values == serial.values
Test.@test default_backend.values == serial.values
