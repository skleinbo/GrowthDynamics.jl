### Pretty printing for various objects.

## TumorConfiguration
Base.show(io::IO, C::GrowthDynamics.TumorConfigurations.TumorConfiguration) = begin
    println(io, typeof(C.lattice))
    println(io, length(C.meta.genotypes),"\tgenotypes")
    print(io, TumorObservables.total_population_size(C),"\tpopulation")
end
