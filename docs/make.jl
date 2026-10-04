using Documenter: Documenter
using StructureFunctions: StructureFunctions

Documenter.DocMeta.setdocmeta!(
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

Documenter.makedocs(;
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
        "Overview and installation" => "index.md",
        "Getting started" => "getting_started.md",
        "Mathematical definitions" => "theory.md",
        "Data and geometry" => "data.md",
        "Execution backends" => "backends.md",
        "Calculations" => "examples.md",
        "Scientific analysis" => ["spectra.md", "khm.md"],
        "Validation and performance" => "validation.md",
        "API reference" => ["api/operators.md", "api/calculations.md", "api/results.md", "api/helpers.md", "api/internals.md"],
        "Contributor notes" => ["architecture.md", "gpu.md", "extensions.md", "digitize.md"],
    ],
    warnonly = false,
    checkdocs = :exports,
)

if get(ENV, "DOCUMENTER_DEPLOY", "false") == "true"
    Documenter.deploydocs(;
        repo = "github.com/jbphyswx/StructureFunctions.jl",
        devbranch = "main",
        push_preview = true,
    )
end
