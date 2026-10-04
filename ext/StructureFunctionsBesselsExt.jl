module StructureFunctionsBesselsExt

using Bessels: Bessels
using StructureFunctions: Calculations as SFC

@inline SFC.bessel_kernel(::Val{0}, x) = Bessels.besselj0(x)
@inline SFC.bessel_kernel(::Val{1}, x) = Bessels.besselj1(x)
@inline SFC.bessel_kernel(::Val{2}, x) = Bessels.besselj(2, x)
@inline SFC.bessel_kernel(::Val{3}, x) = Bessels.besselj(3, x)

# The plane's isotropic kernel.
@inline SFC.isotropic_kernel(::Val{2}, x) = SFC.bessel_kernel(Val(0), x)

SFC.gamma(x) = Bessels.gamma(x)

end # module
