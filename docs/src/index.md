# GrowthDynamics.jl

This package implements various models of growing and evolving populations, both with and without spatial structure.

Furthermore it defines a number of useful observables to be evaluated during simulation.

Finally it contains logic to define and process simulation jobs (module `SimulationRunner`). This functionality however is currently under development and breaking changes are to be expected. The rest of the package may be used quite independently.

Let's quickly give an overview of the components acting together.

## States
A state is a structure comprised of a lattice (see Lattices), each entry representing a member of the population, its value representing its genotype, and various metadata. In unstructered populations the lattice is simply a dummy.

The metadata store information about the current state of the various genotypes, like number of individuals present, their fitness, and so on. Furthermore a phylogenetic tree is recorded during simulation, enabling access to observables like most-recent common ancestors.

__Example:__

```@example 1
import GrowthDynamics.TumorConfigurations # hide
using GrowthDynamics.LatticeTumorDynamics # hide
using GrowthDynamics.TumorObservables # hide
import DataFrames: first # hide

state = TumorConfigurations.uniform_circle(128, 1/10, 0, 1)
```

prepares a state on a two-dimensional hexagonal lattice of size `128^2` that is unoccupied (genotype `0` is per definition to be understood as unoccupied.) except for a centered disk of genotype `1` that comprises `~1/10` of the total population.


## Dynamics

__Example (cont'd)__
```@example 1
die_or_proliferate!(state;
  T=128^2,
  mu=1e0,
  d=1/100,
  fitness=g->1.0 + 0.1*randn()
)

show(state)
```

## Observables

__Example (cont'd)__
```@example 1
# number of polymorphisms and diversity
npolymorphisms(state), mean_pairwise(state)
```
```@example 1
# alleles with frequency larger 0.01
first(allele_spectrum(state, threshold=0.01), 6)
```
