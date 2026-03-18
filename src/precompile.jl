PrecompileTools.@compile_workload begin
    # Common types
    FTs = [Float64, Float32]
    dims = [1, 2, 3]

    for FT in FTs
        bins = [(FT(0), FT(1)), (FT(1), FT(2))]
        sfs = [
            LongitudinalSecondOrderStructureFunction,
            TransverseSecondOrderStructureFunction,
            DiagonalConsistentThirdOrderStructureFunction,
        ]

        for N in dims
            # Tuple API Precompilation
            x_tuple = ntuple(_ -> FT[0, 1, 2], Val{N}())
            u_tuple = ntuple(_ -> FT[0, 1, 2], Val{N}())

            for sf in sfs
                calculate_structure_function(
                    sf,
                    x_tuple,
                    u_tuple,
                    bins;
                    verbose = false,
                    show_progress = false,
                )
            end

            # Matrix API Precompilation
            x_mat = zeros(FT, N, 3)
            u_mat = zeros(FT, N, 3)
            for sf in sfs
                calculate_structure_function(
                    sf,
                    x_mat,
                    u_mat,
                    bins;
                    verbose = false,
                    show_progress = false,
                )
            end
        end

        # Spectral Analysis Precompilation
        x_spec = ntuple(_ -> FT[0, 1], Val{2}())
        u_spec = ntuple(_ -> FT[0, 1], Val{2}())
        ms = (2, 2)
        calculate_spectrum(
            DirectSumBackend(),
            x_spec,
            u_spec,
            ms;
            domain_size = (FT(10), FT(10)),
        )
    end
end
