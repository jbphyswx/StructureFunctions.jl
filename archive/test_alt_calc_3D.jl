using StaticArrays

include("/home/jbenjami/Research_Schneider/CliMA/StructureFunctions.jl/src/StructureFunctions.jl")
create_svector(x) = Main.StructureFunctions.HelperFunctions.create_svector(x)

np = 2e3 - Int(1)
a  = create_svector( rand(Float32,  Int(np)));
b = Vector(a);
nt = 3
aaa  = repeat(a, 1, nt ,2);

f = Tuple(create_svector.(zip(a,a)));

fff = Tuple( Base.Iterators.Generator( x->collect(zip(x...)), collect(zip(eachrow.(eachslice(aaa, dims=3))...)) ));
fff = Tuple( Base.Iterators.Generator( x->collect(zip(x...)), reduce(hcat, eachrow.(eachslice(aaa, dims=3))) ));


nbins=50

# structure_function_type = Main.StructureFunctions.StructureFunctionTypes.SecondOrderStructureFunction()
# structure_function_type = Main.StructureFunctions.StructureFunctionTypes.DiagonalInconsistentThirdOrderStructureFunction()
structure_function_type = Main.StructureFunctions.StructureFunctionTypes.DiagonalConsistentThirdOrderStructureFunction()

show_progress = true
return_sums_and_counts = true
check_nan = true
threaded = false
combine_third_dimension = false
verbose = true
bin_spacing = :logarithmic
nbins = 50

distance_bins = @time Main.StructureFunctions.AlternateCalculationsD.calculate_distance_bins(f; n_distance_bins=nbins, show_progress=show_progress, verbose=verbose, bin_spacing=bin_spacing)
out = @time Main.StructureFunctions.AlternateCalculations3D.calculate_structure_function(f, fff, distance_bins, structure_function_type; show_progress=show_progress, verbose=verbose, return_sums_and_counts=return_sums_and_counts, check_nan=check_nan, threaded=threaded, combine_third_dimension=combine_third_dimension)

# out = @time Main.StructureFunctions.AlternateCalculations3D.calculate_structure_function(f, fff, nbins, structure_function_type; show_progress=show_progress, verbose=verbose, bin_spacing=bin_spacing, return_sums_and_counts=return_sums_and_counts, check_nan=check_nan, threaded=threaded, combine_third_dimension=combine_third_dimension)



# distance_bins = @time Main.StructureFunctions.AlternateCalculations3D.calculate_distance_bins(f; n_distance_bins=nbins, show_progress=show_progress, verbose=verbose, bin_spacing=bin_spacing)


f = Vector(create_svector.(zip(a,a))); # slow?
fff = (x->collect(zip(x...))).( collect(zip(eachrow.(eachslice(aaa, dims=3))...)) );


distance_bins = @time Main.StructureFunctions.AlternateCalculations3D_Union_Types.calculate_distance_bins(f; n_distance_bins=nbins, show_progress=show_progress, verbose=verbose, bin_spacing=bin_spacing)
out = @time Main.StructureFunctions.AlternateCalculations3D_Union_Types.calculate_structure_function(f, fff, distance_bins, structure_function_type; show_progress=show_progress, verbose=verbose, return_sums_and_counts=return_sums_and_counts, check_nan=check_nan, threaded=threaded, combine_third_dimension=combine_third_dimension)
# out = @time Main.StructureFunctions.AlternateCalculations3D_Union_Types.calculate_structure_function(f, fff, 50, structure_function_type; show_progress=show_progress, verbose=verbose, return_sums_and_counts=return_sums_and_counts, check_nan=check_nan, threaded=threaded, combine_third_dimension=combine_third_dimension)



