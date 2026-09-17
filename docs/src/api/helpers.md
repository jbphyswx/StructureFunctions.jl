```@meta
CurrentModule = StructureFunctions
```

# Bins, tapers, fields and geometry

The bin-edge wrappers with `O(1)` digitizing, the tapers shared by the spectra and the harmonic
route, the `HarmonicNodes` and `ModeBinEdges` objects a kernel-binned result carries, the `Fields`
multi-field, and the geometry helpers: pair frames, transverse conventions and the spherical
metric.

```@index
Pages = ["helpers.md"]
```

## Top-level

```@autodocs
Modules = [StructureFunctions]
Private = false
Order   = [:type, :function, :constant]
```

## MultiFields

```@autodocs
Modules = [StructureFunctions.MultiFields]
Private = false
```

## Geometry

```@autodocs
Modules = [StructureFunctions.HelperFunctions]
Private = false
```
