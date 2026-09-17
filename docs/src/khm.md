# Exact laws (KHM)

```@meta
CurrentModule = StructureFunctions
```

The Kármán–Howarth–Monin exact laws relate a third-order structure function in the inertial range to
a dissipation or flux. `StructureFunctions.KHM` inverts them on already-binned results and reports
the residual of each law. Every law is stated for one specific moment, and the laws are not
interchangeable: applying the four-fifths law to `S3SF` returns a number that is wrong by exactly
`5/3` and looks entirely reasonable.

| law | moment | operator | relation | inversion |
|---|---|---|---|---|
| four-fifths (Kolmogorov 1941) | ``⟨δu_L³⟩`` | `L3SFType` | ``⟨δu_L³⟩ = -\tfrac{4}{5} ε r`` | [`KHM.epsilon_from_four_fifths`](@ref) |
| four-thirds | ``⟨δu_L ‖δu‖²⟩`` | `S3SFType` | ``⟨δu_L ‖δu‖²⟩ = -\tfrac{4}{3} ε r`` | [`KHM.epsilon_from_four_thirds`](@ref) |
| Yaglom | ``⟨δu_L (δθ)²⟩`` | `MixedSFType{1,0,2}` | ``⟨δu_L (δθ)²⟩ = -\tfrac{4}{3} ε_θ r`` | [`KHM.epsilon_theta_from_yaglom`](@ref) |

```julia
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

res = SFC.calculate_structure_function(SFT.L3SFType(), x, u, bins)
r = SF.midpoints(res.distance)
ε = SF.KHM.epsilon_from_four_fifths(r, res.values)          # one estimate per bin; flat in the inertial range
SF.KHM.four_fifths_residual(r, res.values, ε[5])              # how far each bin is from the law at that ε
```

The residual of the planar isotropy relation between the longitudinal and transverse second-order
functions, ``D_{TT} = D_{LL} + r\,\mathrm{d}D_{LL}/\mathrm{d}r`` in two dimensions (Lindborg 1999,
eq. 53), is [`KHM.transverse_incompressibility_residual`](@ref); on a sphere it holds only to
``O((r/R)^2)``.

These are inertial-range relations for homogeneous isotropic turbulence with a single cascade. A
synthetic Gaussian field has no cascade and its third-order moments are consistent with zero, so the
laws have nothing to recover there; the spectral flux relations of the transforms
(`spectral_flux`, `enstrophy_flux`) and the regularised fits (`fit_flux`) are the scale-resolved
counterparts.

## Functions

```@autodocs
Modules = [StructureFunctions.KHM]
Private = false
```
