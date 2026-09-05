module StructureFunctionsBesselsExt

using Bessels: Bessels
using StructureFunctions: Calculations as SFC

# Each order is its own method, so every one is strictly more specific than the core fallback and
# adds to it rather than replacing it.
@inline SFC.bessel_kernel(::Val{0}, x) = Bessels.besselj0(x)
@inline SFC.bessel_kernel(::Val{1}, x) = Bessels.besselj1(x)
@inline SFC.bessel_kernel(::Val{2}, x) = Bessels.besselj(2, x)
@inline SFC.bessel_kernel(::Val{3}, x) = Bessels.besselj(3, x)

# The plane's isotropic kernel. One and three dimensions are elementary and live in core, so loading
# this package is what a two-dimensional transform needs and nothing else does.
@inline SFC.isotropic_kernel(::Val{2}, x) = SFC.bessel_kernel(Val(0), x)

end # module
