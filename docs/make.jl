using Documenter
using StructureFunctions

DocMeta.setdocmeta!(
    StructureFunctions,
    :DocTestSetup,
    :(using StructureFunctions);
    recursive = true,
)

const MODULES = [
    StructureFunctions,
    StructureFunctions.StructureFunctionTypes,
    StructureFunctions.StructureFunctionObjects,
    StructureFunctions.Calculations,
]

makedocs(;
    modules = MODULES,
    authors = "Jordan Benjamin",
    sitename = "StructureFunctions.jl",
    format = Documenter.HTML(;
        canonical = "https://jbphyswx.github.io/StructureFunctions.jl",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
        size_threshold = 400 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Backends" => "backends.md",
        "GPU Acceleration" => "gpu.md",
        "Extensions" => "extensions.md",
        "Examples" => "examples.md",
        "Binning Internals" => "uniform_bin_digitize.md",
        "API Reference" => "api.md",
    ],
    # Loose hand-written pages contain links to source files outside docs/src/ and a few
    # not-yet-cross-referenced names; keep the build green and tighten incrementally.
    warnonly = true,
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/jbphyswx/StructureFunctions.jl",
    devbranch = "main",
    push_preview = true,
)
