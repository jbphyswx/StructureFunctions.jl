# A simple script for updating the manifest
# files in all of our environments.
#
# To update dependencies, run:
# julia -e 'using Pkg; Pkg.add("PkgDevTools"); using PkgDevTools; PkgDevTools.update_deps(".")'

using Pkg: Pkg

using PkgDevTools: PkgDevTools

root = dirname(@__DIR__)

PkgDevTools.update_deps(root; auto_all = true)

