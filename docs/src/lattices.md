# [Lattices](@id lattices)

```@meta
    CurrentModule = GrowthDynamics.Lattices
```

Each agent occupies a site on a lattice. They provide a mapping between _coordinates_ and _indices_ of the underlying array.

N.B.: The provided structures are not true lattices, since they do not extend arbitrarily, but have finite size.

Currently, the following lattice types are provided

* [`LineLattice`](@ref)
* [`HexagonalLattice`](@ref)
* [`CubicLattice`](@ref)
* [`FCCLattice`](@ref)

These are subtypes of `RealLattice`. Additionally, a `NoLattice` type is provided for situations without a spatial structure.

!!! warning

    Avoid manipulating the `data` field of a lattice directly.
    Doing so easily leads to an inconsistent state between lattice and
    meta data.

    Use the getter and setter methods for [`TumorConfiguration`](@ref) instead.

## Lattice Types

```@docs
    CubicLattice
    FCCLattice
    HexagonalLattice
    LineLattice
```

Each lattice type derives from `AbstractLattice` and implements the following methods

## Common Methods

```@docs
    coord
    index
    coordination
    dimension
    midpoint
    midpointcoord
    neighbors
    neighbors!
    nneighbors
    density
```


