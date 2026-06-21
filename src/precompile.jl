PrecompileTools.@compile_workload begin
    # Common types
    FTs = [Float64, Float32]
    dims = [2, 3]

    for FT in FTs
        bins = [FT(0), FT(1), FT(2)]
        sfs = [
            LongitudinalSecondOrderStructureFunction,
            TransverseSecondOrderStructureFunction,
            DiagonalConsistentThirdOrderStructureFunction,
        ]

        for N in dims
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
    end
end
