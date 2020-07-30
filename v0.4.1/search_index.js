var documenterSearchIndex = {"docs":
[{"location":"dynamics.html#Available-dynamics-1","page":"Dynamics","title":"Available dynamics","text":"","category":"section"},{"location":"dynamics.html#","page":"Dynamics","title":"Dynamics","text":"Routines to evolve a given state a number of timesteps.","category":"page"},{"location":"dynamics.html#","page":"Dynamics","title":"Dynamics","text":"All routines take a state::TumorConfiguration as their first argument, and further parameters like mutation rate as keyword arguments.","category":"page"},{"location":"dynamics.html#","page":"Dynamics","title":"Dynamics","text":"LatticeTumorDynamics.moran!\nLatticeTumorDynamics.die_or_proliferate!\nLatticeTumorDynamics.exponential!","category":"page"},{"location":"dynamics.html#GrowthDynamics.LatticeTumorDynamics.moran!","page":"Dynamics","title":"GrowthDynamics.LatticeTumorDynamics.moran!","text":"moran!(state::NoLattice{Int}; <keyword arguments>)\n\n(Extended) Moran dynamics on an unstructured population. Grow until carrying capacity is reach. After that individuals begin replacing each other.\n\nArguments\n\nT::Int: the number of steps to advance.\nfitness: function that assigns a fitness value to a genotype g::Int.\np_grow=1.0: Probability with which to actually proliferate. If no proliferation happens, mutation might still occur.\nmu: mutation rate.\nmu_type=[:poisson, :fixed]: Number of mutations is fixed, or Poisson-distributed.\ngenome_length=10^9: Length of the haploid genome.\nd: death rate.\nbaserate: progressing real time is measured in 1/baserate.\nprune_period: prune the phylogeny periodically after no. of steps.\nprune_on_exit: prune before leaving the simulation loop.\ncallback: function of state and time to be called at each iteration.   Used primarily for collecting observables during the run.\nabort: condition on state and time under which to end the run.\n\n\n\n\n\n","category":"function"},{"location":"dynamics.html#GrowthDynamics.LatticeTumorDynamics.die_or_proliferate!","page":"Dynamics","title":"GrowthDynamics.LatticeTumorDynamics.die_or_proliferate!","text":"die_or_proliferate!(state::RealLattice{Int}; <keyword arguments>)\n\nMoran-like dynamics on an spatially structured population. Each step is either a death or (potential) birth and mutation event.\n\nIndividuals die at a rate d. Birthrate depends linearily on the number of neighbors.\n\nArguments\n\nT::Int: the number of steps to advance.\nfitness: function that assigns a fitness value to a genotype. Takes arguments (state, old genotype, new_genotype).\np_grow=1.0: Probability with which to actually proliferate. If no proliferation happens, mutation might still occur.\nmu=0.0: mutation rate.\nmu_type=[:poisson, :fixed]: Number of mutations is fixed, or Poisson-distributed.\ngenome_length=10^9: Length of the haploid genome.\nd=0.0: death rate. Zero halts the dynamics after carrying capacity is reached.\nbaserate=1.0: progressing real time is measured in 1/baserate.\nprune_period=0: prune the phylogeny periodically after no. of steps.\nprune_on_exit=true: prune before leaving the simulation loop.\ncallback: function of state and time to be called at each iteration.   Used primarily for collecting observables during the run.\nabort: condition on state and time under which to end the run.\n\n\n\n\n\n","category":"function"},{"location":"dynamics.html#GrowthDynamics.LatticeTumorDynamics.exponential!","page":"Dynamics","title":"GrowthDynamics.LatticeTumorDynamics.exponential!","text":"exponential!(state::NoLattice{Int}; <keyword arguments>)\n\nRun exponential growth on an unstructered population.\n\nArguments\n\nT::Int: the number of steps (generations) to advance.\nfitness: function that assigns a fitness value to a genotype g::Int.\nmu: mutation rate.\nbaserate: progressing real time is measured in 1/baserate.\nprune_period: prune the phylogeny periodically after no. of steps.\nprune_on_exit: prune before leaving the simulation loop.\ncallback: function of state and time to be called at each iteration.   Used primarily for collecting observables during the run.\nabort: condition on state and time under which to end the run.\n\n\n\n\n\n","category":"function"},{"location":"observables.html#Observables-1","page":"Observables","title":"Observables","text":"","category":"section"},{"location":"observables.html#","page":"Observables","title":"Observables","text":"Modules = [TumorObservables]","category":"page"},{"location":"observables.html#GrowthDynamics.TumorObservables.allele_fractions","page":"Observables","title":"GrowthDynamics.TumorObservables.allele_fractions","text":"Dictionary of (SNP, freq).\n\n\n\n\n\n","category":"function"},{"location":"observables.html#GrowthDynamics.TumorObservables.allele_spectrum-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.allele_spectrum","text":"allele_spectrum(state;[ threshold=0.0, read_depth=total_population_size(state)])\n\nReturn a DataFrame with count, frequency of every polymorphism. Additionally sample from the population.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.common_snps-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.common_snps","text":"List polymorphisms that are common to all genotypes.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.has_children-Tuple{Any,Any}","page":"Observables","title":"GrowthDynamics.TumorObservables.has_children","text":"Does a genotype have any children?\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.mean_pairwise-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.mean_pairwise","text":"Diversity (mean pairwise difference of mutations) of a population.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.nchildren-Tuple{TumorConfiguration,Any}","page":"Observables","title":"GrowthDynamics.TumorObservables.nchildren","text":"Number of direct descendends of a genotype.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.npolymorphisms-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.npolymorphisms","text":"npolymorphisms(S::TumorConfiguration)\n\nNumber of polymrphisms\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.pairwise-Tuple{TumorConfiguration,Any,Any}","page":"Observables","title":"GrowthDynamics.TumorObservables.pairwise","text":"pairwise(S::TumorConfiguration, i, j)\n\nNumber of pairwise genomic differences between genotype indices i,j.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.pairwise-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.pairwise","text":"pairwise(S::TumorConfiguration)\n\nMatrix of pairwise differences.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.polymorphisms-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.polymorphisms","text":"polymorphisms(S::TumorConfiguration)\n\nVector of polymorphisms (segregating sites).\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.population_size-Tuple{TumorConfiguration,Any}","page":"Observables","title":"GrowthDynamics.TumorObservables.population_size","text":"Dictionary (genotype, population size)\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.sampled_allele_fractions","page":"Observables","title":"GrowthDynamics.TumorObservables.sampled_allele_fractions","text":"sampled_allele_fractions(S::TumorConfiguration[, t=0, samples=length(S.meta.npops)])\n\nRandomly sample genotypes(!) and calculate frequencies of contained SNPs. Return a dictionary (SNP, freq).\n\n\n\n\n\n","category":"function"},{"location":"observables.html#GrowthDynamics.TumorObservables.total_population_size-Tuple{TumorConfiguration}","page":"Observables","title":"GrowthDynamics.TumorObservables.total_population_size","text":"Total population size. Duh.\n\n\n\n\n\n","category":"method"},{"location":"observables.html#GrowthDynamics.TumorObservables.allele_size","page":"Observables","title":"GrowthDynamics.TumorObservables.allele_size","text":"Dictionary (SNP, population count)\n\n\n\n\n\n","category":"function"},{"location":"observables.html#GrowthDynamics.TumorObservables.tajimasd-Tuple{Any,Any,Any}","page":"Observables","title":"GrowthDynamics.TumorObservables.tajimasd","text":"See https://en.wikipedia.org/wiki/Tajima%27s_D\n\n\n\n\n\n","category":"method"},{"location":"index.html#GrowthDynamics.jl-1","page":"Overview","title":"GrowthDynamics.jl","text":"","category":"section"},{"location":"index.html#","page":"Overview","title":"Overview","text":"This package implements various models of growing and evolving populations, both with and without spatial structure.","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Furthermore it defines a number of useful observables to be evaluated during simulation.","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Finally it contains logic to define and process simulation jobs (module SimulationRunner). This functionality however is currently under development and breaking changes are to be expected. The rest of the package may be used quite independently.","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Let's quickly give an overview of the components acting together.","category":"page"},{"location":"index.html#States-1","page":"Overview","title":"States","text":"","category":"section"},{"location":"index.html#","page":"Overview","title":"Overview","text":"A state is a structure comprised of a lattice (see Lattices), each entry representing a member of the population, its value representing its genotype, and various metadata. In unstructered populations the lattice is simply a dummy.","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"The metadata store information about the current state of the various genotypes, like number of individuals present, their fitness, and so on. Furthermore a phylogenetic tree is recorded during simulation, enabling access to observables like most-recent common ancestors.","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Example:","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"import GrowthDynamics.TumorConfigurations # hide\nusing GrowthDynamics.LatticeTumorDynamics # hide\nusing GrowthDynamics.TumorObservables # hide\nimport DataFrames: first # hide\n\nstate = TumorConfigurations.uniform_circle(128, 1/10, 0, 1)","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"prepares a state on a two-dimensional hexagonal lattice of size 128^2 that is unoccupied (genotype 0 is per definition to be understood as unoccupied.) except for a centered disk of genotype 1 that comprises ~1/10 of the total population.","category":"page"},{"location":"index.html#Dynamics-1","page":"Overview","title":"Dynamics","text":"","category":"section"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Example (cont'd)","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"die_or_proliferate!(state;\n  T=128^2,\n  mu=1e0,\n  d=1/100,\n  fitness=g->1.0 + 0.1*randn()\n)\n\nshow(state)","category":"page"},{"location":"index.html#Observables-1","page":"Overview","title":"Observables","text":"","category":"section"},{"location":"index.html#","page":"Overview","title":"Overview","text":"Example (cont'd)","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"# number of polymorphisms and diversity\nnpolymorphisms(state), mean_pairwise(state)","category":"page"},{"location":"index.html#","page":"Overview","title":"Overview","text":"# alleles with frequency larger 0.01\nfirst(allele_spectrum(state, threshold=0.01), 6)","category":"page"}]
}
