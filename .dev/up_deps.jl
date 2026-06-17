# A simple script for updating the manifest
# files in all of our environments.
#
# NOTE: This script is DISCONTINUED in favor of PkgDevTools.
# To update dependencies, run:
# julia -e 'using Pkg; Pkg.add("PkgDevTools"); using PkgDevTools; PkgDevTools.update_deps(".")'

using Pkg: Pkg

using PkgDevTools: PkgDevTools

root = dirname(@__DIR__)

PkgDevTools.update_deps(root; auto_all = true)

# root = dirname(@__DIR__)
# dirs = (
#     root,
#     joinpath(root, "test"),
#     joinpath(root, ".dev"),
# )

# for dir in dirs
#     Pkg.activate(dir)
#     Pkg.update()
# end
