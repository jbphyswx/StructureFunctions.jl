```@meta
CurrentModule = StructureFunctions
```

# Operators

An operator names the statistic of a pair's increment that is accumulated. Every pairwise operator is
callable on one pair, `sf(δu, r̂)`, and the transforms consume it through its polynomial contract
(`moment_contract`). The shorthands `L2SFType`, `T2SFType`, `S3SFType`, … are re-exported from the
top-level module; the instances `L2SF`, `T2SF`, … are the same operators already constructed.

```@index
Pages = ["operators.md"]
```

```@autodocs
Modules = [StructureFunctions.StructureFunctionTypes]
Private = false
Order   = [:type, :constant, :function]
```
