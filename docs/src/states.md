# States

A _state_ is a collection of

* either a `lattice<:RealLattice` if the model is spatial (see [lattices](@ref)), or a dummy placeholder `NoLattice` if it is not.
* a phylogenetic tree, which is a directed graph where the root(s) represent the wildtype(s).
* metadata about the population and its genetics. See [Metadata](@ref).
* the time the state has been evolved for by invoking dynamics on it.

States are of type [`Population`](@ref).

It is normally not required to manipulate metadata, phylogeny or lattice directly. Convenient methods to manipulate state are provided.

```@setup getset
using Graphs
using GrowthDynamics
using GrowthDynamics.Populations
using GrowthDynamics.Lattices
```

```@repl getset
    state, _ = uniform(CubicLattice, 32; g=0)

    # fails because the new genotype is yet unknown
    state[16,16,16] = 1

    # make it known first
    push!(state, 1)
    state[16,16,16] = 1;

    state.meta

    state[15:17,15:17,15:17]

    push!(state, 2)
    state[1:size(state,1), 1, 1] .= 2;

    state.meta
```

## Convenience constructors

Some convenience methods for common geometries are provided

```@docs
    uniform
    spheref
    spherer
    single_center
    half_space
    sphere_with_single_mutant_on_outer_shell
```

## API

```@docs
    Population
```
