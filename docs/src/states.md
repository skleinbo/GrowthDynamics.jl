# States

A _state_ is a collection of

* either a `lattice<:RealLattice` if the model is spatial, or a dummy placeholder `NoLattice` if it is not.
* a phylogenetic tree which is a directed graph where the root(s) are the wildtype(s).
* meta data about the population and its genetics. See @ref(MetaData).
* the time the state has been evolved for by invoking dynamics on it.

States are of type `TumorConfiguration`.
