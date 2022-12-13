### Pretty printing for various objects.

## Population
Base.show(io::IO, C::GrowthDynamics.Populations.Population) = begin
    println(io, typeof(C.lattice))
    println(io, length(C.meta),"\tgenotypes")
    print(io, TumorObservables.total_population_size(C),"\tpopulation")
end
