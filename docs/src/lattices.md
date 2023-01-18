# [Lattices](@id lattices)

Each cell occupies a site on a lattice. A lattice provides a mapping between _coordinates_ and _indices_ of the underlying array.

**N.B.**: The structures are not true lattices, because they do not extend to infinity, but have finite size.

Currently, the following lattices are provided

* [`LineLattice`](@ref)
* [`HexagonalLattice`](@ref)
* [`CubicLattice`](@ref)
* [`FCCLattice`](@ref)

They are subtypes of `RealLattice`. Populations without spatial structure "live" on a [`NoLattice`](@ref).

!!! warning

    Avoid manipulating the `data` field of a lattice directly.
    Doing so easily leads to an inconsistent state between lattice and
    metadata.

    Use the getter and setter methods for [`Population`](@ref) instead.

## Lattice Types

```@docs
    CubicLattice
    FCCLattice
    HexagonalLattice
    LineLattice
    NoLattice
```

Each lattice type except `NoLattice` derives from `RealLattice` and implements the following methods

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

## Radial geometries

Populations often grow with approximate radial symmetry. One might then be interested in for example
grouping cells with respect to their distance from an origin.

```@docs
    shell
    isonshell
    conicsection
```
