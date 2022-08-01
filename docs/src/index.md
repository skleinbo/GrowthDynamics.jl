# GrowthDynamics.jl

This package implements various models of growing and evolving populations, both with and without spatial structure.

A number of useful observables are defined.

## Quick Start

Let's give an overview of the components acting together.

## States

A state is a structure comprised of a lattice (see Lattices), each entry representing a member of the population, its value representing its genotype, and various metadata. In unstructered populations the lattice is simply a dummy.

The metadata store information about the current state of the various genotypes, like number of individuals present, their fitness, and so on. Furthermore a phylogenetic tree is recorded during simulation, enabling access to observables like most-recent common ancestors, or tracking of lineages.

```@example 1
import GrowthDynamics.TumorConfigurations: spheref
import GrowthDynamics.Lattices: HexagonalLattice
using GrowthDynamics.LatticeTumorDynamics
using GrowthDynamics.TumorObservables
import DataFrames: first # hide

state, _ = spheref(HexagonalLattice, 128, f=1/10, g1=0, g2=1)
```

prepares a state on a two-dimensional hexagonal lattice of size `128^2` that is unoccupied (genotype `0` is per definition understood as unoccupied.) except for a centered disk of genotype `1` that comprises `~1/10` of the total population.

Every mutation event introduces a new genotype. They are by default labeled consecutively by integers, but custom labels are possible.

## Dynamics

Now evolve the population.

```@example 1
eden_with_density!(state;
  T=128^2, # timesteps
  mu=1e0,  # mutation rate per genome (not site!)
  d=1/100, # death rate
  fitness=(s,g_old,g_new)->1.0 + 0.1*randn() # function to assign fitness to new genotypes
)

show(state)
```

```@example 1
using CairoMakie #hide
hist(state.meta[:, :fitnesses], axis=(xlabel="Fitness", ylabel="Frequency")) #hide
```

## Observables

```@example 1
# number of polymorphisms and diversity
npolymorphisms(state), mean_pairwise(state)
```

```@example 1
# alleles with frequency larger 0.01
first(sort(allele_spectrum(state, threshold=0.01), :fpop), 6)
```
