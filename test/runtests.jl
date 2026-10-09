using Test: Test

const TEST_FILES = (
    "test_result_buffers.jl", "test_helpers.jl", "test_bin_edges.jl",
    "test_bin_constructors.jl", "test_core_correctness.jl", "test_cpu_pair_blocking.jl", "test_gridded.jl",
    "test_gridded_flowgeometries.jl", "test_gridded_fft.jl", "test_gridded_masked.jl", "test_gridded_moments.jl",
    "test_gridded_zonal.jl", "test_gridded_separable.jl", "test_gridded_device.jl", "test_gridded_batch.jl",
    "test_gridded_single_pass.jl", "test_gridded_weights.jl", "test_pair_weights.jl", "test_scattered_modes.jl",
    "test_sorted_line.jl", "test_spectra_lagspace.jl", "test_harmonic_sphere.jl", "test_directional.jl",
    "test_multifields.jl", "test_operator_contract.jl", "test_pair_value.jl", "test_single_pass.jl",
    "test_single_pass_2d.jl", "test_e2e.jl", "test_stability.jl", "test_shorthands.jl", "test_shape_contract.jl",
    "test_spherical_geometry.jl", "test_tensor_khm.jl", "test_known_truth.jl", "test_transforms.jl", "test_fits.jl",
    "test_fit_numerics.jl", "test_cpu_workspace.jl", "test_no_silent_fallback.jl", "test_allocations.jl",
    "test_batch_matrix.jl", "test_2d_binning.jl", "test_inplace.jl", "test_device.jl", "test_gpu_in_range.jl",
    "test_gpu_script_hygiene.jl", "test_parallel_equivalence.jl", "test_mpi.jl", "test_aqua.jl", "test_jet.jl",
)

# Each file in its own module, named after the file, so no file sees another's definitions; each file's test set is
# outermost, so its summary prints when the file finishes, and a file with failures does not stop the others.
const FAILED = String[]
for file in TEST_FILES
    name, path = Symbol(first(splitext(file))), joinpath(@__DIR__, file)
    try
        Test.@testset "$file" begin
            @eval Main module $name
                include($path)
            end
        end
    catch err
        err isa Test.TestSetException || rethrow()
        push!(FAILED, file)
    end
end
isempty(FAILED) || error("test files with failures or errors: $(join(FAILED, ", "))")
