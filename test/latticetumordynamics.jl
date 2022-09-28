import GrowthDynamics
using GrowthDynamics.LatticeTumorDynamics
import GrowthDynamics.TumorObservables: total_population_size

@testset "die_or_proliferate" begin
    μ = 0e-2
    L = 2^6
    N = L^3
    Nfinal=2^10
    fit_func(s, gold, gnew) = gnew==2 ? 1.3 : 1.0
    
    tumor = GrowthDynamics.TumorConfigurations.single_center(Lattices.CubicLattice, L, g1=0, g2=1)[1]
    eden_with_density!(tumor;DEBUG=false,
        fitness=fit_func,
        d=0.0,
        T=2N,
        mu=μ,
        prune_on_exit=true,
        det_growth=false,
        det_mutations=false,
        abort=s->total_population_size(s) >= Nfinal || s.t >= Nfinal
    )
    @test total_population_size(tumor) == Nfinal
end


@testset "moran" begin
    μ = 0e-2
    L = 2^6
    N = L^3
    Nfinal=2^10
    fit_func(s, gold, gnew) = gnew==2 ? 1.3 : 1.0
    
    tumor = GrowthDynamics.TumorConfigurations.nolattice_state()[1]
    moran!(tumor;DEBUG=false,
        fitness=fit_func,
        d=0.0,
        T=2N,
        mu=μ,
        prune_on_exit=true,
        det_growth=false,
        det_mutations=false,
        K=Nfinal÷2,
        abort=s->total_population_size(s) >= Nfinal || s.t >= Nfinal
    )
    @test total_population_size(tumor) == Nfinal÷2
end