module Lattices

export  AbstractLattice,
        RealLattice,
        NoLattice,
        HexagonalLattice,
        LineLattice,
        HCPLattice,
        out_of_bounds,
        ind2coord,
        coord2ind,
        neighbours,
        neighbours!,
        LineLatticeNeighbours,
        HexLatticeNeighbours,
        visualize

import MetaGraphs: MetaDiGraph

abstract type AbstractLattice end
abstract type AbstractLattice1D{T} <:AbstractLattice end
abstract type AbstractLattice2D{T} <:AbstractLattice end
abstract type AbstractLattice3D{T} <:AbstractLattice end
const RealLattice{T} = Union{AbstractLattice1D{T}, AbstractLattice2D{T}, AbstractLattice3D{T}}

out_of_bounds(I::CartesianIndex,N) = mapreduce(m->m<1||m>N,|,Tuple(I))

function latticevectors(L::AbstractLattice3D)
    org = ind2coord(L, 0,0,0)
    (ind2coord(L, 1,0,0).-org, ind2coord(L, 0,1,0).-org, ind2coord(L, 0,0,1).-org)
end

function latticevectors(L::AbstractLattice2D)
    org = ind2coord(L, 0,0)
    (ind2coord(L, 1,0).-org, ind2coord(L, 0,1).-org)
end

const Neighbours{dim} = Vector{CartesianIndex{dim}}

## --- NO LATTICE --- ##
mutable struct NoLattice <: AbstractLattice
    N::Int # System size
end


## --- BEGIN 1D Lattice OBC -- ##
mutable struct LineLattice{T} <: AbstractLattice1D{T}
    Na::Int # Lattice sites in direction a
    a::Real # Lattice constant
    data::Array{T,1}
end

LineLatticeNeighbours() = [ CartesianIndex(0) for _ in 1:2 ]

neighbours(L::LineLattice, m) = begin
    tmp = LineLatticeNeighbours()
    neighbours!(tmp, m, L)
    tmp
end

neighbours!(nn::Neighbours{1}, I::CartesianIndex, L::LineLattice) = @inbounds begin
        m = I[1]
        nn[1] = CartesianIndex(m-1)
        nn[2] = CartesianIndex(m+1)
end


## --- END 1D Lattice OBC -- ##




## --- BEGIN Hexagonal lattice -- ##
mutable struct HexagonalLattice{T} <: AbstractLattice2D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    a::Real # Lattice constant
    data::Array{T,2}
end

# const HexLatticeNeighbours = Neighbours{2}
HexLatticeNeighbours() = [ CartesianIndex(0,0) for _ in 1:6 ]


ind2coord(L::HexagonalLattice{T}, I::CartesianIndex) where T<:Any = L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2])
function coord2ind(L::T, x,y) where T<:HexagonalLattice
    n = round(Int, 2/sqrt(3)*y/(L.a))
    m = round(Int, (x/L.a-n/2))
    return CartesianIndex(m,n)
end

neighbours(L::HexagonalLattice, I::CartesianIndex) = begin
    tmp = HexLatticeNeighbours()
    neighbours!(tmp,I,L)
    tmp
end

neighbours!(nn::Neighbours{2}, I::CartesianIndex, L::HexagonalLattice) = @inbounds begin
    m,n = Tuple(I)
    if isodd(n)
        nn[1] = CartesianIndex(m-1, n-1)
        nn[2] = CartesianIndex(m-1, n+1)
        nn[5] = CartesianIndex(m-1, n)
        nn[6] = CartesianIndex(m+1, n)
        nn[3] = CartesianIndex(m, n+1)
        nn[4] = CartesianIndex(m, n-1)
    else
        nn[1] = CartesianIndex(m, n-1)
        nn[2] = CartesianIndex(m, n+1)
        nn[5] = CartesianIndex(m-1, n)
        nn[6] = CartesianIndex(m+1, n)
        nn[3] = CartesianIndex(m+1, n+1)
        nn[4] = CartesianIndex(m+1, n-1)
    end
end
## -- END HexagonalLattice -- ##

mutable struct HCPLattice{T} <: AbstractLattice3D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    Nc::Int # Lattice sites in direction c
    a::Real # Lattice constant
    data::Array{T,3}
    Phylogeny::MetaDiGraph
end
HCPNeighbours() = [ CartesianIndex(0,0) for _ in 1:12 ]

neighbours!(nn::Neighbours{3}, I::CartesianIndex, L::HCPLattice) = @inbounds begin
    m,n,l = Tuple(I)
    if isodd(n)
        nn[1] = CartesianIndex(m-1, n-1, l)
        nn[2] = CartesianIndex(m-1, n+1, l)
        nn[5] = CartesianIndex(m-1, n, l)
        nn[6] = CartesianIndex(m+1, n, l)

        nn[3] = CartesianIndex(m, n+1, l)
        nn[4] = CartesianIndex(m, n-1, l)

        nn[7] = CartesianIndex(m-1, n-1, l+1)
        nn[8] = CartesianIndex(m, n-1, l+1)
        nn[9] = CartesianIndex(m, n, l+1)

        nn[10] = CartesianIndex(m-1, n-1, l-1)
        nn[11] = CartesianIndex(m, n-1, l-1)
        nn[12] = CartesianIndex(m, n, l-1)
    else
        nn[1] = CartesianIndex(m, n-1, l)
        nn[2] = CartesianIndex(m, n+1, l)
        nn[5] = CartesianIndex(m-1, n, l)
        nn[6] = CartesianIndex(m+1, n, l)
        nn[3] = CartesianIndex(m+1, n+1, l)
        nn[4] = CartesianIndex(m+1, n-1, l)

        nn[7] = CartesianIndex(m, n-1, l+1)
        nn[8] = CartesianIndex(m+1, n-1, l+1)
        nn[9] = CartesianIndex(m, n, l+1)

        nn[10] = CartesianIndex(m, n-1, l-1)
        nn[11] = CartesianIndex(m+1, n-1, l-1)
        nn[12] = CartesianIndex(m, n, l-1)
    end
end
neighbours(L::HCPLattice{<:Any}, I::CartesianIndex) = begin
    tmp = HCPNeighbours()
    neighbours!(tmp,I,L)
    tmp
end

ind2coord(L::HCPLattice{T}, I::CartesianIndex) where T<:Any = L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2],I[3])

## -- END HCPLattice -- ##


end
