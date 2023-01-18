module LatticeDynamics

import Distributions: Binomial, Exponential, cdf
import Graphs: nv, vertices, add_vertex!, add_edge!
using ..Lattices
import Random: shuffle!
using StatsBase: Weights, sample, mean
import ..Phylogenies: sample_ztp
import ..Populations: Population, add_genotype!, add_snps!, annotate_snps!, getfitness
import ..Populations: connect!, index, hassnps, lastgenotype, prune_phylogeny!, _resize!, rename!, snpsfrom
import ..Observables: total_population_size
import WeightedSampling: adjust_weight!, WeightedSampler, sample as smp, weight

export eden_with_density!, exponential!, moran!, twonew!

@enum Action none=0 proliferate=1 mutate=2 die=3

const MutationProfile = Tuple{Symbol,Float64,Int} # (:poisson/:fixed, rate, genome length)

function mu_func(mu)
    if mu isa Real
        function(args...)
            (mu, 1.0 - exp(-mu))
        end
    else
        function(args...)
            _mu = mu(args...)
            (_mu, 1.0-exp(-_mu))
        end
    end
end

###--- Start of simulation methods ---###

include("latticedynamics/edenwithdensity.jl")
include("latticedynamics/exponential.jl")
include("latticedynamics/moran.jl")
include("latticedynamics/twonew.jl")

const dynamics_dict = Dict(
    :moran => moran!,
    :eden_with_density => eden_with_density!,
)

## -- END module LatticeDynamics --
end
