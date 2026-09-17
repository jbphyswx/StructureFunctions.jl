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
    StructureFunctions.HelperFunctions,
    StructureFunctions.MultiFields,
    StructureFunctions.KHM,
]

makedocs(;
    modules = MODULES,
    authors = "Jordan Benjamin",
    sitename = "StructureFunctions.jl",
    format = Documenter.HTML(;
        canonical = "https://jbphyswx.github.io/StructureFunctions.jl",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
        size_threshold_warn = 400 * 1024,
        size_threshold = 800 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Backends" => "backends.md",
        "GPU Acceleration" => "gpu.md",
        "Extensions" => "extensions.md",
        "Examples" => "examples.md",
        "Walkthrough" => "walkthrough.md",
        "Exact Laws (KHM)" => "khm.md",
        "Validation" => "validation.md",
        "Binning Internals" => "uniform_bin_digitize.md",
        "API Reference" => [
            "api/operators.md",
            "api/calculations.md",
            "api/results.md",
            "api/helpers.md",
            "api/internals.md",
        ],
    ],
    warnonly = false,
    checkdocs = :exports,
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/jbphyswx/StructureFunctions.jl",
        devbranch = "main",
        push_preview = true,
    )
end
