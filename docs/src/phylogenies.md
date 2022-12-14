# Phylogenies

A `state::Population` has a phylogenetic tree and metadata about the genotypes attached.  
Any time evolution routine should take care of maintaining this tree and metadata.

* The vertices `v` are always numbered from 1 and correspond to the genotype `meta[v]`. Use `index`(@ref) to find the id of a genotype.
* Inserting a new genotype into the population via `push!(::Population, newgenotype)` will add a vertex to the phylogeny, but won't create an edge that connects it to the tree.

!!! warning
    Use `add_edge!(state, newgenotype, parent)` (preferred) or `add_edge!(state.phylogeny, newgenotype_id, parent_id)` to connect the new genotype to the phylogeny.

```@autodocs
    Modules = [GrowthDynamics.Phylogenies]
```

## Pruning

In the course of a simulation run, especially with large mutation rates, the phylogenetic tree and metadata can become quite extensive. They contain an entry for every genotype that ever was in existence, thus representing the full history of the population. Eventually performance will deteriorate. If the full tree is not needed, one may opt to occasionally _prune_ the tree and metadata, that is, remove internal nodes that have died out.

See also the keyword arguments `prune_period` and `prune_on_exit`.

```@docs
prune_phylogeny!
```
