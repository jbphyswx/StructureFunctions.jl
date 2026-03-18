using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# NOTE: This script is DISCONTINUED in favor of the official JuliaFormatter CLI
# or using julia -e 'using JuliaFormatter; format(".")'.
# This project now uses a root-level .JuliaFormatter.toml.

using JuliaFormatter: JuliaFormatter

JuliaFormatter.format(".")
