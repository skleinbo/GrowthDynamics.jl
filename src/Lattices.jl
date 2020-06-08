module Lattices

export  AbstractLattice,
        RealLattice,
        NoLattice,
        HexagonalLattice,
        hex_nneighbors,
        line_nneighbors,
        LineLattice,
        CubicLattice,
        HCPLattice,
        out_of_bounds,
        ind2coord,
        coord2ind,
        neighbors,
        neighbors!,
        LineLatticeNeighbors,
        HexLatticeNeighbors,
        nneighbors,
        empty_neighbors,
        density!

import Base.Iterators: product
import LinearAlgebra: norm, normalize, cross, det, dot
using StaticArrays

import LightGraphs: SimpleDiGraph

include("Geometry.jl")

# Lattice type structure
abstract type AbstractLattice end
abstract type AbstractLattice1D{T} <:AbstractLattice end
abstract type AbstractLattice2D{T} <:AbstractLattice end
abstract type AbstractLattice3D{T} <:AbstractLattice end


## --- NO LATTICE --- ##
## T is the type of genotypes we will store.
mutable struct NoLattice{T} <: AbstractLattice
    N::Int # System size
end
"Dummy 'lattice' for systems without spatial structure. Carrying capacity `N`."
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

empty_neighbors(L::RealLattice) = [ CartesianIndex(zeros(Int64, dimension(L))...) for _ in 1:coordination(L)]


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

"""
Distance between two points on a line lattice.
"""
function dist(L::Lattices.LineLattice, I::CartesianIndex, J::CartesianIndex)
    dx = L.a * ( J[1]-I[1] )
    return abs(dx)
end

## --- END 1D Lattice OBC -- ##




## --- BEGIN Hexagonal lattice -- ##
mutable struct HexagonalLattice{T} <: AbstractLattice2D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    a::Real # Lattice constant
    data::Array{T,2}
end

## NOTE: This implementation uses offset-coordinates by default
##       where every _even_ row is offset by +1/2 lattice spacing.
##

"""
Convert offset to cube coordinates.
"""
function offset_to_cube(L::Lattices.HexagonalLattice, I::CartesianIndex{2})
    x = I[2] - ( I[1] + ( I[1]&1)) / 2
    z = I[1]
    y = -x-z
    return x, y, z
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
        nn[5] = CartesianIndex(m+1, n-1)
        nn[6] = CartesianIndex(m, n-1)
        nn[2] = CartesianIndex(m-1, n)
        nn[4] = CartesianIndex(m+1, n)
        nn[3] = CartesianIndex(m, n+1)
    else
        nn[6] = CartesianIndex(m, n-1)
        nn[1] = CartesianIndex(m-1, n)
        nn[5] = CartesianIndex(m+1, n)
        nn[2] = CartesianIndex(m-1, n+1)
        nn[3] = CartesianIndex(m, n+1)
        nn[4] = CartesianIndex(m+1, n+1)
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

"""
Distance between two points on a hex lattice.
"""
function euclidean_dist(L::Lattices.HexagonalLattice, I::CartesianIndex, J::CartesianIndex)
    Ic = offset_to_cube(L, I)
    Jc = offset_to_cube(L, J)
    return sqrt( (Ic[1]-Jc[1])^2 + (Ic[2]-Jc[2])^2 + (Ic[3]-Jc[3])^2)
    # # cos(pi/6) is the angle between
    # # y-axis and (1,1)-direction
    # dy = L.a * (J[2] - I[2])*cos(pi/6)
    # # Every odd step in y-direction implies half a step
    # # in x-direction
    # dx = L.a * ( J[1]-I[1] - 1/2 * float(isodd(J[2]-I[2])) )
    #
    # return sqrt(dx^2+dy^2)
end

"""
Manhatten distance on the hex lattice.
Useful for determining rings.
"""
function manhatten_dist(L::Lattices.HexagonalLattice, I::CartesianIndex, J::CartesianIndex)
    I = offset_to_cube(L, I)
    J = offset_to_cube(L, J)

    return ( abs(I[1]-J[1]) + abs(I[2]-J[2]) + abs(I[3]-J[3]) ) / 2
end


## -- END HexagonalLattice -- ##

## -- BEGIN CubicLattice -- ##
mutable struct CubicLattice{T} <: AbstractLattice3D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    Nc::Int # Lattice sites in direction c
    a::Real # Lattice constant
    data::Array{T,3}
end

CubicLatticeNeighbors() = [ CartesianIndex(0,0,0) for _ in 1:6 ]

neighbors!(nn::Neighbors{3}, I::CartesianIndex, L::CubicLattice) = @inbounds begin
    m,n,l = Tuple(I)
    nn[1] = CartesianIndex(m-1, n, l)
    nn[2] = CartesianIndex(m+1, n, l)
    nn[3] = CartesianIndex(m, n-1, l)
    nn[4] = CartesianIndex(m, n+1, l)
    nn[5] = CartesianIndex(m, n, l-1)
    nn[6] = CartesianIndex(m, n, l+1)
end

neighbors(L::CubicLattice{<:Any}, I::CartesianIndex) = begin
    tmp = CubicLatticeNeighbors()
    neighbors!(tmp,I,L)
    tmp
end

ind2coord(L::CubicLattice{T}, I::CartesianIndex) where T<:Any = L.a .* Tuple(I)

function cubic_nneighbors(I::CartesianIndex, N)
    coord = 6
    if I[1]<=1 || I[1]>=N
        coord -= 1
    end
    if I[2]<=1 || I[2]>=N
        coord -= 1
    end
    if I[3]<=1 || I[3]>=N
        coord -= 1
    end
    coord
end

import LinearAlgebra: norm
function euclidean_dist(L::Lattices.CubicLattice, I::CartesianIndex, J::CartesianIndex)
    Δ = (I - J)
    return L.a * norm( Tuple(Δ) )
end

function intersectsplane(L::Lattices.CubicLattice, P::Plane)
    M = map([SVector{3}(1.0,0.0,0.0), SVector{3}(0.0,1.0,0.0), SVector{3}(0.0,0.0,1.0)]) do x
        A = SMatrix{3,3}([x P.u P.v])
        if det(A) != 0.0
            return (:inv, inv(A))
        else
            return (:ninv, SMatrix{3,3}(P.u*P.u' + P.v*P.v'))
        end
    end
    @show M
    return function(q)
        if euclidean_dist_to_plane(q, P) > 3
            return false
        end
        bIntersects = false
        for m in M
            if m[1] == :inv
                x = m[2]*(q .- P.p)
                if 1/2 <= x[1] < 1/2
                    bIntersects = true
                    break
                end
            else
                bIntersects = (m[2]*(q - P.p) ≈ (q - P.p))
                if bIntersects
                    break
                end
            end
        end
        bIntersects
    end
end

function intersect_lattice_with_plane(L::Lattices.CubicLattice, P::Plane)
    # starting index
    #dp, Ip = findmin(euclidean_dist_matrix(L, P.p))
    indices = product(1:L.Na, 1:L.Na, 1:L.Na)
    ifunc = intersectsplane(L, P)
    BitArray(ifunc(SVector{3}(I)) for I in indices)
end

## -- END CubicLattice -- ##


## -- BEGIN HCPLattice -- ##

mutable struct HCPLattice{T} <: AbstractLattice3D{T}
    Na::Int # Lattice sites in direction a
    Nb::Int # Lattice sites in direction b
    Nc::Int # Lattice sites in direction c
    a::Real # Lattice constant
    data::Array{T,3}
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

function hcp_nneighbors(I::CartesianIndex, N)
    coord_xy = hex_nneighbors(I, N)

    if 2 <= I[3]
        coord_xy += 1
    end
    if I[3] < N
        coord_xy += 1
    end
    coord_xy
end


"""
Distance between two points on a HCP lattice.
"""
function euclidean_dist(L::Lattices.HCPLattice, I::CartesianIndex, J::CartesianIndex)
    # cos(pi/6) is the angle between
    # y-axis and (1,1)-direction
    dy = L.a * (J[2] - I[2])*cos(pi/6)
    # Every odd step in y-direction implies half a step
    # in x-direction
    dx = L.a * ( J[1]-I[1] - 1/2 * float(isodd(J[2]-I[2])) )
    dz = L.a *(J[3] - I[3])
    return sqrt(dx^2+dy^2+dz^2)
end

## -- END HCPLattice --##

dimension(::NoLattice) = 0
dimension(::AbstractLattice1D) = 1
dimension(::AbstractLattice2D) = 2
dimension(::AbstractLattice3D) = 3

coordination(::LineLattice) = 2
coordination(::HexagonalLattice) = 6
coordination(::CubicLattice) = 6
coordination(::HCPLattice) = 12

## Generic nearest neighbor functions
for (L,short) in collect(zip([:LineLattice, :HexagonalLattice, :CubicLattice], [:line, :hex, :cubic]))
    eval(
    quote
        nneighbors(l::$L,n,N) = Lattices.$(Symbol(short,"_nneighbors"))(n,N)
    end
    )
end


## Define density function for different lattice types.
for LatticeType in [Lattices.LineLattice, Lattices.HexagonalLattice, Lattices.CubicLattice]
    if LatticeType <: Lattices.HexagonalLattice
        nn_function = :hex_nneighbors
    elseif LatticeType <: Lattices.LineLattice
        nn_function = :line_nneighbors
    elseif LatticeType <: Lattices.CubicLattice
        nn_function = :cubic_nneighbors
    end
    eval(
    quote
        function density!(nn,L::$LatticeType,ind::CartesianIndex)
            lin_N = size(L.data, 1)
            neighbors!(nn, ind, L)
            tot = $(nn_function)(ind,lin_N)
            nz =  count(x->!out_of_bounds(x,lin_N) && L.data[x]!=0, nn)
            return nz/tot
        end
    end)
end


end
