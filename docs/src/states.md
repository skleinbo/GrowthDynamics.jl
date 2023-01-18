# Populations

A _population_ is a collection of

* either a `lattice<:RealLattice` if the model is spatial (see [lattices](@ref)), or a dummy placeholder `NoLattice` if it is not.
* a phylogenetic tree, which is a directed graph where the root(s) represent the wild type(s).
* metadata about the population and its genetics. See [Metadata](@ref).
* the time the state has been evolved for by invoking dynamics on it.

They are of type [`Population`](@ref).

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
    # last paremeter is the parental genotype
    # because there is none yet, set it to `nothing`
    add_genotype!(state, 1, nothing)
    state[16,16,16] = 1;

    state.meta

    state[15:17,15:17,15:17]

    push!(state, 2)
    state[1:size(state,1), 1, 1] .= 2;

    state.meta
```

## Convenience constructors

A handful of methods to generate common geometries are provided. They return a tuple
(`population`,`auxilliary`). The latter holds information that might be nice to have. It is often `nothing`, but for example [`spherer`](@ref) returns a vector with all indices contained in the ball.

```@docs
    uniform
    spheref
    spherer
    single_center
    half_space
    sphere_with_single_mutant_on_outer_shell
    sphere_with_diverse_outer_shell
```

An initial population without spatial structure can be constructed with

```@docs
nolattice_state
```

## API

### Manipulating genotypes

```@docs
    Population
    add_genotype!
    remove_genotype!
    push!
```

### Methods to add mutations

```@docs
add_snps!
annotate_snps!
annotate_lineage!
```
