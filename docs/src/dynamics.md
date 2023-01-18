# Evolving populations

Routines to advance a population state for a number of time steps, or until
a pre-specified condition is met.

## Available dynamics

All routines take a `Population` as their first and only positional argument, and
further parameters like mutation rate as keyword arguments.

## Spatial

### An Eden-like model

```@docs
LatticeDynamics.eden_with_density!
```

## Non-spatial

```@docs
LatticeDynamics.moran!
LatticeDynamics.exponential!
```

## Adding your own

There is no particular interface or signature an evolution routine must adhere to.

However, to avoid putting the lattice, metadata and phylogeny in an inconsistent state, a few tips should be followed

* Use the getter and setter methods for [`MetaData`](@ref) instead of manipulating the fields directly.
* Using `metadata[g=genotype; Val(:field)]` is more performant than `metadata[g=genotype; :field]` if `:field` is constant, because
    the former avoids dynamic dispatch.
* Use `population[index] (= genotype)` to get/set the genotype of a cell at a position.
    Do not manipulate `population.lattice.data` directly.
* Prefer [`add_genotype!`](@ref) over [`push!`](@ref).
* Use [`remove_genotype!`](@ref).
* For performance reasons SNPs are either `Vector{Int}` or `nothing`. Check for the latter with [`hassnps`](@ref) before
   adding new ones.
* Not a must, but don't forget to advance the step/real time counters `population.t` and `population.treal` after each simulation step.
