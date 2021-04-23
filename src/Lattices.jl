module Lattices

export  AbstractLattice,
        RealLattice,
        NoLattice,
        HexagonalLattice,
        LineLattice,
        CubicLattice,
        HCPLattice,
        out_of_bounds,
        coord,
        index,
        neighbors,
        neighbors!,
        nneighbors,
        empty_neighbors,
        density,
        density!,
        dist

import Base: size, length, getindex, setindex!, maybeview, firstindex, lastindex
import Base.Iterators: product
import LinearAlgebra: norm, normalize, cross, det, dot
using StaticArrays

include("Geometry.jl")

# Lattice type structure
abstract type AbstractLattice end
abstract type AbstractLattice1D{T} <:AbstractLattice end
abstract type AbstractLattice2D{T} <:AbstractLattice end
abstract type AbstractLattice3D{T} <:AbstractLattice end


## --- NoLattice --- ##
## T is the index type for genotypes.
"""
    NoLattice{T}
Dummy 'lattice' for systems without spatial structure. Carrying capacity `N`."
"""
struct NoLattice{T} <: AbstractLattice end

NoLattice() = NoLattice{Int64}()
size(L::NoLattice) = ()
size(L::NoLattice, i...) = 0
length(L::NoLattice) = 0
## -- END NoLattice

const RealLattice{T} = Union{AbstractLattice1D{T}, AbstractLattice2D{T}, AbstractLattice3D{T}}
const TypedLattice{T} = Union{RealLattice{T}, NoLattice{T}}

const Neighbors{dim} = Vector{CartesianIndex{dim}}

## Indexing
for method in [:maybeview, :getindex, :setindex!, :firstindex, :lastindex]
    eval(quote
        Base.$(method)(L::RealLattice, args...) = $(method)(L.data, args...)
    end)
end

function out_of_bounds(I::CartesianIndex, N)
    I = Tuple(I)
    oob = false
    for i in 1:length(I)
        if I[i]<1 || I[i]>N[i]
            oob = true
            break
        end
    end
    oob
end

nneighbors(L::RealLattice, I) = nneighbors(typeof(L), size(L), I)

LatticeNeighbors(L::RealLattice) = LatticeNeighbors(dimension(L), coordination(L))
LatticeNeighbors(dim, coordination) = [ CartesianIndex(zeros(Int64, dim)...) for _ in 1:coordination ]


## size & lengths dispatch on the data field of any RealLattice ##
"""
    size(L::RealLattice, [i])
Shape of `L.data` (in direction `i`).
"""
size(L::RealLattice) = size(L.data)
size(L::RealLattice, i...) = size(L.data, i...)

"""
    length(L::RealLattice)
Number of elements of `L.data`.
"""
length(L::RealLattice) = length(L.data)

"""
    neighbors(L::RealLattice, I)

Vector of neighbors of index I. Returns Array{CartesianIndex{dimension(L)}}.
Does not check for bounds.
"""
function neighbors(L::RealLattice, I)
    tmp = LatticeNeighbors(L)
    neighbors!(tmp, L, I)
    tmp
end

"""
    dist(::RealLattice, I, J)

Euclidean distance between two points on a lattice.
"""
function dist(L::RealLattice, I, J)
    dx = coord(L, I) .- coord(L, J)
    return norm(dx)
end

"""
    function euclidean_dist_matrix(L, p)

Matrix similar to `L.data` filled with euclidean distances wrt. the point `p`.
"""
euclidean_dist_matrix(L::RealLattice, p::CartesianIndex) = euclidean_dist_matrix(L, coord(L,p))
function euclidean_dist_matrix(L::RealLattice, p)
    N = size(L)
    map(CartesianIndices(L.data)) do I
        dist(L, coord(L, I), p)
    end
end


## --- BEGIN 1D Lattice OBC -- ##
struct LineLattice{T} <: AbstractLattice1D{T}
    a::Float64 # Lattice constant
    data::AbstractArray{T, 1}
end

coord(L::LineLattice{T}, I) where T = L.a * I[1]

index(L::LineLattice{T}, x) where T = CartesianIndex(round(Int64, x/L.a))

"""
In-place version of `neighbors`.
"""
function neighbors!(nn::Neighbors{1}, ::LineLattice, I)
    @inbounds begin
        m = I[1]
        nn[1] = CartesianIndex(m-1)
        nn[2] = CartesianIndex(m+1)
    end
end

function nneighbors(::Type{LineLattice{T}}, N, I) where T
    nn = 2
    if I[1] == 1 || I[1] == N[1]
        nn -= 1
    end
    return nn
end


## --- END 1D Lattice OBC -- ##




## --- BEGIN Hexagonal lattice -- ##
struct HexagonalLattice{T} <: AbstractLattice2D{T}
    a::Real # Lattice constant
    data::AbstractArray{T,2}
end

## NOTE: This implementation uses offset-coordinates by default
##       where every _even_ row is offset by +1/2 lattice spacing.
##
## Some methods concerning distances utilize "cube coordinates"
## (see https://www.redblobgames.com/grids/hexagons/#coordinates)

"""
Convert offset to cube coordinates.
"""
function offset_to_cube(::Lattices.HexagonalLattice, I)
    x = I[2] - ( I[1] + ( I[1]&1)) / 2
    z = I[1]
    y = -x-z
    return x, y, z
end

function coord(L::HexagonalLattice, I)
    x = (I[1]-1)*L.a
    y = (I[2]-1)*L.a/√3
    if iseven(I[2])
        x += 1/2*L.a
    end
    x,y
end

function index(L::T, p) where T<:HexagonalLattice
    x,y, = p
    n = round(Int, y*√3/L.a) + 1 
    if iseven(n)
        x -= 1/2*L.a
    end
    m = round(Int, x/L.a+1)
    return CartesianIndex(m,n)
end

function neighbors!(nn::Neighbors{2}, L::HexagonalLattice, I)
    @inbounds begin
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
end

function nneighbors(::Type{HexagonalLattice{T}}, N, I) where T
    if isodd(I[1])
        if (I[1] == 1 || I[1] == N[1]) && 2<=I[2]<N[2]
            return 4
        elseif I[2] == 1 && 2<=I[1]<N[1]
            return 3
        elseif I[2] == N && 2<=I[1]<N[1]
            return 5
        elseif (I[1] == N[1] && I[2] == N[2]) || (I[1] == 1 && I[2] == 1)
            return 2
        elseif I[1] == 1 && I[2] == N[2]
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
Manhatten distance on the hex lattice.
"""
function manhatten_dist(L::HexagonalLattice, I, J)
    I = offset_to_cube(L, I)
    J = offset_to_cube(L, J)

    return ( abs(I[1]-J[1]) + abs(I[2]-J[2]) + abs(I[3]-J[3]) ) / 2
end

## -- END HexagonalLattice -- ##

## -------------------------- ##

## -- BEGIN CubicLattice --   ##
struct CubicLattice{T} <: AbstractLattice3D{T}
    a::Real # Lattice constant
    data::AbstractArray{T,3}
end
CubicLattice(L::Integer) = CubicLattice(1.0, fill(0, L,L,L))

coord(L::CubicLattice{T}, I) where T<:Any = L.a .* (Tuple(I) .- 1)

function index(L::CubicLattice, p)
    return  round.(Int, p ./ L.a)
end

neighbors!(nn::Neighbors{3}, L::CubicLattice, I) = @inbounds begin
    m,n,l = Tuple(I)
    nn[1] = CartesianIndex(m-1, n, l)
    nn[2] = CartesianIndex(m+1, n, l)
    nn[3] = CartesianIndex(m, n-1, l)
    nn[4] = CartesianIndex(m, n+1, l)
    nn[5] = CartesianIndex(m, n, l-1)
    nn[6] = CartesianIndex(m, n, l+1)
end

function nneighbors(::Type{CubicLattice{T}}, N, I) where T
    coord = 6
    if I[1]<=1 || I[1]>=N[1]
        coord -= 1
    end
    if I[2]<=1 || I[2]>=N[2]
        coord -= 1
    end
    if I[3]<=1 || I[3]>=N[3]
        coord -= 1
    end
    coord
end

## Special routines for the CubicLattice outside the RealLattice interface ##

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

function isonplane(L::Lattices.CubicLattice, P::Plane)
    pts = product(1:L.Na, 1:L.Na, 1:L.Na)
    BitArray( euclidean_dist(SVector{3}(q), P)<L.a for q in pts )
end

## -- END CubicLattice -- ##


## -- BEGIN HCPLattice -- ##

struct HCPLattice{T} <: AbstractLattice3D{T}
    a::Float64
    b::Float64
    data::AbstractArray{T,3}
end

coord(L::HCPLattice, I::CartesianIndex) = L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2],I[3])
## TODO: index(::HCPLattice)
function index(L::HCPLattice)
end

function neighbors!(nn::Neighbors{3}, ::HCPLattice, I)
    @inbounds begin
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
end

function nneighbors(::Type{HCPLattice{T}}, N, I) where T
    coord_xy = nneighbors(HexagonalLattice, N[1:2], I[1:2])

    if 2 <= I[3]
        coord_xy += 1
    end
    if I[3] < N
        coord_xy += 1
    end
    coord_xy
end
## -- END HCPLattice --##

dimension(::NoLattice) = 0
dimension(::AbstractLattice1D) = 1
dimension(::AbstractLattice2D) = 2
dimension(::AbstractLattice3D) = 3

coordination(L::RealLattice) = coordination(typeof(L))
coordination(::Type{LineLattice{T}}) where T = 2
coordination(::Type{HexagonalLattice{T}}) where T = 6
coordination(::Type{CubicLattice{T}}) where T = 6
coordination(::Type{HCPLattice{T}}) where T = 12

"""
    density(L::RealLattice, I)

Calculates `(occupied sites)/(no. of neighbors)`

"Occupied" means != zero(eltype(L.data))
"""
density(L::RealLattice, I) = density!(LatticeNeighbors(L), L, I)
function density!(nn::Neighbors, L::RealLattice, I)
    tot = nneighbors(L, I)
    neighbors!(nn, L, I)
    nz =  count(x->!out_of_bounds(x, size(L)) && L.data[x]!=0, nn) ## TODO: makes assumption about the numerical value that repr. unoccupied
    return nz/tot
end


## END MODULE ##
end
