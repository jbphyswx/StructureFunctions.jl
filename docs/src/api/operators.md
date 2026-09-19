```@meta
CurrentModule = StructureFunctions
```

# Operators

An operator names the statistic of a pair's increment that is accumulated. Every pairwise operator is
callable on one pair, `sf(δu, r̂)`, and the transforms consume it through its polynomial contract
(`moment_contract`). The operators live in `StructureFunctions.StructureFunctionTypes` and are
reached through it — `SFT.L2SFType()` after `using StructureFunctions: StructureFunctionTypes as
SFT` — rather than from the top-level module, which exports only the bin edges, tapers, nodes and
result containers. The shorthands `L2SFType`, `T2SFType`, `S3SFType`, … name the types; the
instances `L2SF`, `T2SF`, … are the same operators already constructed.

```@index
Pages = ["operators.md"]
```

```@autodocs
Modules = [StructureFunctions.StructureFunctionTypes]
Private = false
Order   = [:type, :constant, :function]
```
