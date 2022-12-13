using GrowthDynamics
using .Lattices
using .Populations
using BenchmarkTools

μ = 0e-2
L = 2^7
N = L^3
Nfinal=10_000

fit_func(s, gold, gnew) = gnew==2 ? 1.3 : 1.0

state_spatial = GrowthDynamics.Populations.single_center(Lattices.CubicLattice, L, g1=0, g2=1)

suite = BenchmarkGroup()

suite["dynamics"] = BenchmarkGroup(["dynamics"])
suite["dynamics"]["dop"] = BenchmarkGroup(["gillespie", "die_or_proliferate"])
suite["tc"] = BenchmarkGroup()
suite["tc"]["density"] = BenchmarkGroup(["neighbors, density"])
suite["tc"]["getindex"] = BenchmarkGroup(["density","arraylike"])
suite["tc"]["setindex"] = BenchmarkGroup(["density","arraylike"])
suite["tc"]["getindexarray"] = BenchmarkGroup(["density","arraylike","comparison"])
suite["tc"]["meta"] = BenchmarkGroup()
suite["tc"]["meta"]["push"] = BenchmarkGroup()

for L in [32,64,128]
    suite["dynamics"]["dop"][string(L)] = @benchmarkable LatticeTumorDynamics.die_or_proliferate!(state_spatial;DEBUG=false,
            fitness=fit_func,
            d=0.0,
            T=2N,
            mu=μ,
            prune_on_exit=true,
            det_growth=false,
            det_mutations=false,
            abort=s->total_population_size(s) >= Nfinal || s.t >= Nfinal
        ) setup=begin
            μ = 0e-2; N=$L^3; Nfinal=10_000
            state_spatial=single_center(CubicLattice, $L, g1=0, g2=1)
        end evals=2 samples=10 seconds=90
    
    suite["tc"]["density"][string(L)] = @benchmarkable begin
        Lattices.density!(nn, state.lattice, I)
    end setup=(nn = Lattices.LatticeNeighbors(3,6); state = uniform(CubicLattice, $L; g=1); I = Tuple(rand(1:$L,3)) )
    suite["tc"]["getindex"][string(L)] = @benchmarkable begin
        @inbounds state[I]
    end setup=(state = uniform(CubicLattice, $L; g=1); I = CartesianIndex(rand(1:$L,3)...) )
    suite["tc"]["setindex"][string(L)] = @benchmarkable begin
        @inbounds state[I] = 2
    end setup=(state = uniform(CubicLattice, $L; g=1); I = CartesianIndex(rand(1:$L,3)...) )
    suite["tc"]["getindexarray"][string(L)] = @benchmarkable begin
        @inbounds state[I]
    end setup=(state = rand($L,$L,$L); I = CartesianIndex(rand(1:$L,3)...) )
    ## Crashes Julia 1.6.1
    suite["tc"]["meta"]["push"][string(L)] = @benchmarkable begin
        push!(state.meta, 2)
    end setup=(state = uniform(CubicLattice, $L; g=1)) evals=1
    
end
