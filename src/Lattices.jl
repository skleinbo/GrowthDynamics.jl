module Lattices

export  AbstractLattice,
        RealLattice,
        NoLattice,
        HexagonalLattice,
        hex_nneighbors,
        line_nneighbors,
        LineLattice,
        HCPLattice,
        out_of_bounds,
        ind2coord,
        coord2ind,
        neighbors,
        neighbors!,
        LineLatticeNeighbors,
        HexLatticeNeighbors,
        nneighbors

import LightGraphs: SimpleDiGraph

abstract type AbstractLattice end
abstract type AbstractLattice1D{T} <:AbstractLattice end
abstract type AbstractLattice2D{T} <:AbstractLattice end
abstract type AbstractLattice3D{T} <:AbstractLattice end

## --- NO LATTICE --- ##
## T is the type of genotypes we will store.
mutable struct NoLattice{T} <: AbstractLattice
    N::Int # System size
end
NoLattice(N::Int) = NoLattice{Int64}(N)

const RealLattice{T} = Union{AbstractLattice1D{T}, AbstractLattice2D{T}, AbstractLattice3D{T}}
const AnyTypedLattice{T} = Union{RealLattice{T}, NoLattice{T}}

out_of_bounds(I::CartesianIndex,N) = mapreduce(m->m<1||m>N,|,Tuple(I))

function latticevectors(L::AbstractLattice3D)
    org = ind2coord(L, 0,0,0)
    (ind2coord(L, 1,0,0).-org, ind2coord(L, 0,1,0).-org, ind2coord(L, 0,0,1).-org)
end

function latticevectors(L::AbstractLattice2D)
    org = ind2coord(L, 0,0)
    (ind2coord(L, 1,0).-org, ind2coord(L, 0,1).-org)
end

const Neighbors{dim} = Vector{CartesianIndex{dim}}



## --- BEGIN 1D Lattice OBC -- ##
mutable struct LineLattice{T} <: AbstractLattice1D{T}
    Na::Int # Lattice sites in direction a
    a::Real # Lattice constant
    data::Array{T,1}
end

LineLatticeNeighbors() = [ CartesianIndex(0) for _ in 1:2 ]

ind2coord(L::LineLattice{T}, I::CartesianIndex) where T = L.a * I[1]
coord2ind(L::LineLattice{T}, x) where T = CartesianIndex(round(Int64, x/L.a))


neighbors(L::LineLattice, m) = begin
    tmp = LineLatticeNeighbors()
    neighbors!(tmp, m, L)
    tmp
end

neighbors!(nn::Neighbors{1}, I::CartesianIndex, L::LineLattice) = @inbounds begin
        m = I[1]
        nn[1] = CartesianIndex(m-1)
        nn[2] = CartesianIndex(m+1)
end

function line_nneighbors(I::CartesianIndex, N)
    nn = 2
    if I[1] == 1 || I[1] == N
        nn -= 1
    end
    return nn
end



## --- END 1D Lattice OBC -- ##




## --- BEGIN Hexagonal lattice -- ##
mutable struct HexagonalLattice{T} <: AbstractLattice2D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    a::Real # Lattice constant
    data::Array{T,2}
end

# const HexLattices = Neighbors{2}
HexLatticeNeighbors() = [ CartesianIndex(0,0) for _ in 1:6 ]


ind2coord(L::HexagonalLattice{T}, I::CartesianIndex) where T<:Any = L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2])
function coord2ind(L::T, x,y) where T<:HexagonalLattice
    n = round(Int, 2/sqrt(3)*y/(L.a))
    m = round(Int, (x/L.a-n/2))
    return CartesianIndex(m,n)
end

neighbors(L::HexagonalLattice, I::CartesianIndex) = begin
    tmp = HexLatticeNeighbors()
    neighbors!(tmp,I,L)
    tmp
end

neighbors!(nn::Neighbors{2}, I::CartesianIndex, L::HexagonalLattice) = @inbounds begin
    m,n = Tuple(I)
    if isodd(m)
        nn[1] = CartesianIndex(m-1, n-1)
        nn[2] = CartesianIndex(m-1, n)
        nn[3] = CartesianIndex(m, n+1)
        nn[4] = CartesianIndex(m+1, n)
        nn[5] = CartesianIndex(m+1, n-1)
        nn[6] = CartesianIndex(m, n-1)
    else
        nn[1] = CartesianIndex(m-1, n)
        nn[2] = CartesianIndex(m-1, n+1)
        nn[3] = CartesianIndex(m, n+1)
        nn[4] = CartesianIndex(m+1, n+1)
        nn[5] = CartesianIndex(m+1, n)
        nn[6] = CartesianIndex(m, n-1)
    end
end
function hex_nneighbors(I::CartesianIndex, N)
    if isodd(I[1])
        if (I[1] == 1 || I[1] == N) && 2<=I[2]<N
            return 4
        elseif I[2] == 1 && 2<=I[1]<N
            return 3
        elseif I[2] == N && 2<=I[1]<N
            return 5
        elseif (I[1] == N && I[2] == N) || (I[1] == 1 && I[2] == 1)
            return 2
        elseif I[1] == 1 && I[2] == N
            return 3
        else
            return 6
        end
    else
        if (I[1] == 1 || I[1] == N) && 2<=I[2]<N
            return 4
        elseif I[2] == 1 && 2<=I[1]<N
            return 5
        elseif I[2] == N && 2<=I[1]<N
            return 3
        elseif I[1] == N  && I[2] == N
            return 2
        elseif I[1] == N && I[2] == 1
            return 3
        else
            return 6
        end
    end
end

## -- END HexagonalLattice -- ##

mutable struct HCPLattice{T} <: AbstractLattice3D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    Nc::Int # Lattice sites in direction c
    a::Real # Lattice constant
    data::Array{T,3}
    Phylogeny::SimpleDiGraph
end
HCPNeighbors() = [ CartesianIndex(0,0) for _ in 1:12 ]

neighbors!(nn::Neighbors{3}, I::CartesianIndex, L::HCPLattice) = @inbounds begin
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
neighbors(L::HCPLattice{<:Any}, I::CartesianIndex) = begin
    tmp = HCPNeighbors()
    neighbors!(tmp,I,L)
    tmp
end

ind2coord(L::HCPLattice{T}, I::CartesianIndex) where T<:Any = L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2],I[3])

## -- END HCPLattice -- ##

for (L,short) in collect(zip([:LineLattice, :HexagonalLattice], [:line, :hex]))
    eval(
    quote
        nneighbors(l::$L,n,N) = Lattices.$(Symbol(short,"_nneighbors"))(n,N)
    end
    )
end



end
