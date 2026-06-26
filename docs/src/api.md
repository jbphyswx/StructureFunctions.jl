```@meta
CurrentModule = StructureFunctions
```

# API Reference

Complete reference for the public API. Use the index below to jump to a symbol, or browse the
grouped sections that follow.

```@index
```

## Structure-function operator types

The pairwise operators (`SFT` = `StructureFunctions.StructureFunctionTypes`) select *which*
structure function is accumulated. Shorthands such as `L2SFType`, `T2SFType`, `S3SFType` are
re-exported from the top-level module.

```@autodocs
Modules = [StructureFunctions.StructureFunctionTypes]
Order   = [:type, :function, :constant]
```

## Calculation entry points

The `Calculations` submodule (`SFC`) holds the compute API: the point-field and batched drivers,
the single-pass functions, backend types, and the GPU workspace.

```@autodocs
Modules = [StructureFunctions.Calculations]
Order   = [:function, :type, :constant]
```

## Result containers

Binned results live in `StructureFunctionObjects`. Raw accumulators carry the `…SumsAndCounts`
suffix; the bare names are the averaged/derived views.

```@autodocs
Modules = [StructureFunctions.StructureFunctionObjects]
Order   = [:type, :function]
```

## Bin edges & top-level API

Fast O(1) bin-edge wrappers (`LinearBinEdges`, `LogBinEdges`, `InfPaddedBinEdges`) and the
remaining top-level re-exports.

```@autodocs
Modules = [StructureFunctions]
Order   = [:type, :function, :constant]
```
