# Phylogenies

A `state::TumorConfiguration` has a phylogenetic tree and metadata about the genotypes attached.  
Any time evolution routine should take care of maintaining this tree and metadata.

* The vertices `v` are always numbered from 1 and correspond to the genotype `meta[v]`. Use `index`(@ref) to find the id of a genotype.
* Inserting a new genotype into the population via `push!(::TumorConfiguration, newgenotype)` will add a vertex to the phylogeny, but won't create an edge that connects it to the tree.
  !!! warning
      Use `add_vertex!(state, newgenotype, parent)` (preferred) or `add_vertex!(state.phylogeny, newgenotype_id, parent_id)` to connect the new genotype to the phylogeny.

```@autodocs
    Modules = [GrowthDynamics.Phylogenies]
```
