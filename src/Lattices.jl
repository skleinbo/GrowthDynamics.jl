module Lattices

export  AbstractLattice,
        CubicLattice,
        FCCLattice,
        HCPLattice,
        HexagonalLattice,
        LineLattice,
        NoLattice,
        RealLattice

export  conicsection,
        coord,
        coordination,
        density,
        density!,
        dimension,
        dist,
        index,
        isonshell,
        midpoint,
        midpointcoord,
        neighbors!,
        nneighbors,
        neighbors,
        out_of_bounds,
        radius,
        shell,
        spacings,
        volume

import Base: size, length, getindex, setindex!, maybeview, firstindex, lastindex
import Base.Iterators: product
import CoordinateTransformations: SphericalFromCartesian
import Dictionaries: index
import LinearAlgebra: norm, normalize, cross, det, dot

using GeometryBasics
using Rotations
using StaticArrays

include("Geometry.jl")

const Index{N} = Union{CartesianIndex{N}, NTuple{N, T}} where T

# Lattice type structure
abstract type AbstractLattice{T, N} end


## --- NoLattice --- ##
## T is the index type for genotypes.
"""
    NoLattice{T}
Dummy 'lattice' for systems without spatial structure."
"""
struct NoLattice{T} <: AbstractLattice{T, 0} end

NoLattice() = NoLattice{Int64}()
size(L::NoLattice) = ()
size(L::NoLattice, i...) = 0
length(L::NoLattice) = 0
## -- END NoLattice

const RealLattice{T} = Union{AbstractLattice{T, 1}, AbstractLattice{T, 2}, AbstractLattice{T, 3}}
const TypedLattice{T} = Union{RealLattice{T}, NoLattice{T}}

const Neighbors{z, dim} = MVector{z, CartesianIndex{dim}}

## Indexing
for method in [:maybeview, :getindex, :setindex!, :firstindex, :lastindex]
    eval(quote
        Base.$(method)(L::RealLattice, args...) = $(method)(L.data, args...)
    end)
end

Base.@propagate_inbounds out_of_bounds(L::Lattices.RealLattice, I) = out_of_bounds(I, size(L))

function out_of_bounds(I::CartesianIndex{D}, sz::NTuple{D, <:Integer}) where D
    oob = false
    for i in 1:D
        @inbounds if I[i]<1 || I[i]>sz[i]
            oob = true
            break
        end
    end
    oob
end

@inline out_of_bounds(I::CartesianIndex, N::Integer) = mapreduce(m->m<1||m>N,|,Tuple(I))

@inline nneighbors(L::RealLattice, I) = nneighbors(typeof(L), size(L), I)

Neighbors(L::RealLattice) = Neighbors(dimension(L), coordination(L))
Neighbors(dim, coordination) = Neighbors{coordination, dim}(undef)


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
Base.@propagate_inbounds function neighbors(L::RealLattice, I)
    tmp = Neighbors(L)
    neighbors!(tmp, L, I)
    tmp
end

"""
    midpoint(L)

Return the index of the point nearest to the geometeric center of the lattice `L`.
"""
function midpoint(L::RealLattice)
    index(L, realsize(L)./2)
end

"""
    midpointcoord(L)

Return the coordinate of the point nearest to the geometeric center of the lattice `L`.
"""
midpointcoord(L::RealLattice) = coord(L, midpoint(L))

"""
    dist(::RealLattice, I, J)

Euclidean distance between two indices of a lattice.
"""
function dist(L::RealLattice, I, J)
    dx = coord(L, I) .- coord(L, J)
    return norm(dx)
end

"""
    function euclidean_dist_matrix(L, p)

Matrix similar to `L.data` filled with euclidean distances wrt. the point `p`.
"""
euclidean_dist_matrix(L::RealLattice, p::Index) = euclidean_dist_matrix(L, coord(L,p))
function euclidean_dist_matrix(L::RealLattice, p)
    N = size(L)
    map(CartesianIndices(L.data)) do I
        dist(L, coord(L, I), p)
    end
end

function realsize(L::RealLattice)
    coord(L, size(L)) .- coord(L, first(CartesianIndices(L.data)))
end


####################################
## --- BEGIN Line lattice (1D) -- ##
####################################
struct LineLattice{T, A<:AbstractArray{T, 1}} <: AbstractLattice{T, 1}
    a::Float64 # Lattice constant
    data::A
end

coord(L::LineLattice{T}, I::Index{1}) where T = L.a * I[1]

index(L::LineLattice{T}, x) where T = CartesianIndex(Tuple(round(Int, x/L.a)))

"""
In-place version of `neighbors`.
"""
Base.@propagate_inbounds function neighbors!(nn::Neighbors{2,1}, ::LineLattice, I::Index{1})
    m = I[1]
    nn[1] = CartesianIndex(m-1)
    nn[2] = CartesianIndex(m+1)
end

Base.@propagate_inbounds function nneighbors(::Type{LineLattice{T, A}}, N, I::Index{1}) where {T, A}
    nn = 2
    if I[1] == 1 || I[1] == N[1]
        nn -= 1
    end
    return nn
end
## --- END 1D Lattice OBC -- ##


#########################################
## --- BEGIN Hexagonal lattice (2D) -- ##
#########################################

struct HexagonalLattice{T, A<:AbstractArray{T, 2}} <: AbstractLattice{T, 2}
    a::Float64 # Lattice constant
    data::A
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

function coord(L::HexagonalLattice, I::Index{2})
    x = (I[2]-1)*L.a
    y = (I[1]-1)*L.a*sin(pi/3)
    if iseven(I[1])
        x += 1/2*L.a
    end
    Point2f0(x,y)
end

function index(L::HexagonalLattice, p)
    x,y, = Tuple(p)
    n = round(Int, y*√3/L.a) + 1 
    if iseven(n)
        x -= 1/2*L.a
    end
    m = round(Int, x/L.a+1)
    return CartesianIndex(m,n)
end

Base.@propagate_inbounds function neighbors!(nn::Neighbors{6,2}, ::HexagonalLattice, I)
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

function nneighbors(::Type{HexagonalLattice{T, A}}, N, I) where {T, A}
    if isodd(I[1])
        if (I[1] == 1 || I[1] == N[1]) && 2<=I[2]<N[2]
            return 4
        elseif I[2] == 1 && 2<=I[1]<N[1]
            return 3
        elseif I[2] == N[2] && 2<=I[1]<N[1]
            return 5
        elseif (I[1] == N[1] && I[2] == N[2]) || (I[1] == 1 && I[2] == 1)
            return 2
        elseif I[1] == 1 && I[2] == N[2]
            return 3
        else
            return 6
        end
    else
        if (I[1] == 1 || I[1] == N[1]) && 2<=I[2]<N[2]
            return 4
        elseif I[2] == 1 && 2<=I[1]<N[1]
            return 5
        elseif I[2] == N[2] && 2<=I[1]<N[1]
            return 3
        elseif I[1] == N[1]  && I[2] == N[2]
            return 2
        elseif I[1] == N[1] && I[2] == 1
            return 3
        else
            return 6
        end
    end
end

"""
Manhatten distance between indices on the hex lattice.
"""
function manhatten_dist(L::HexagonalLattice, I, J)
    I = offset_to_cube(L, I)
    J = offset_to_cube(L, J)

    return ( abs(I[1]-J[1]) + abs(I[2]-J[2]) + abs(I[3]-J[3]) ) / 2
end
## -- END HexagonalLattice -- ##


#####################################
## -- BEGIN CubicLattice (3D) --   ##
#####################################

struct CubicLattice{T, A<:AbstractArray{T,3}} <: AbstractLattice{T, 3}
    a::Float64 # Lattice constant
    data::A
end
CubicLattice(L::Integer) = CubicLattice(1.0, fill(0, L,L,L))

coord(L::CubicLattice, I::Index{3})::Point3f0 = L.a .* (Point3f0(Tuple(I)) .- 1)

function index(L::CubicLattice, p)
    return  CartesianIndex(Tuple(round.(Int, p ./ L.a) .+ 1))
end

Base.@propagate_inbounds function neighbors!(nn::Neighbors{6,3}, L::CubicLattice, I)
    m,n,l = Tuple(I)
    nn[1] = CartesianIndex(m-1, n, l)
    nn[2] = CartesianIndex(m+1, n, l)
    nn[3] = CartesianIndex(m, n-1, l)
    nn[4] = CartesianIndex(m, n+1, l)
    nn[5] = CartesianIndex(m, n, l-1)
    nn[6] = CartesianIndex(m, n, l+1)
end

@inline function nneighbors(::Type{CubicLattice{T, A}}, N, I) where {T, A}
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
function nneighbors(::Type{CubicLattice{T, A}}, N::Integer, I) where {T, A}
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

"""
    isonshell(L::CubicLattice, p, r, o)

    Determine whether a lattice point `p` is on a shell with radius `r` wrt.
    the origin `o`. A shell is defined as the collection of points with |(p-o)|≤r+a/2
    where `a` is the lattice spacing.
"""
@inline function isonshell(L::RealLattice, p, r, o=coord(L, midpoint(L)))
    a = spacings(L)[1] / 2

    p′ = p .- o
    r-a < norm(p′) ≤ r+a
end

"""
    shell(L::CubicLattice, r, o=coord(L, midpoint(L)))

Return indices of shell of radius `r` around `o`.
"""
function shell(L::RealLattice, r, o=coord(L, midpoint(L)))
    expected_surface = round(Int, Lattices.volume(r, 3)  -Lattices.volume(r-1, 3))
    out = Vector{CartesianIndex{3}}(undef, expected_surface)
    j = 0
    for I in CartesianIndices(L.data)
        if isonshell(L, coord(L, I), r, o)
            j += 1
            if j<=expected_surface
                @inbounds out[j] = I
            else
                push!(out, I)
            end
        end
    end
    resize!(out, j)
    out
end

## -- END CubicLattice -- ##


###################################
## --- BEGIN HCP lattice (3D) -- ##
###################################

struct HCPLattice{T, A<:AbstractArray{T,3}} <: AbstractLattice{T, 3}
    a::Float64
    b::Float64
    data::A
end

coord(L::HCPLattice, I::Index{3}) = Point3f0(L.a .* (I[1] - 1/2*I[2],sqrt(3)/2*I[2],I[3]))
## TODO: index(::HCPLattice)
function index(L::HCPLattice)
end

Base.@propagate_inbounds function neighbors!(nn::Neighbors{3}, ::HCPLattice, I)
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

function nneighbors(::Type{HCPLattice{T, A}}, N, I) where {T, A}
    coord_xy = @views nneighbors(HexagonalLattice, N[1:2], I[1:2])

    if 2 <= I[3]
        coord_xy += 1
    end
    if I[3] < N
        coord_xy += 1
    end
    coord_xy
end
## -- END HCPLattice --##

###################################
## --- BEGIN FCC lattice (3D) -- ##
###################################

struct FCCLattice{T, A<:AbstractArray{T,3}} <:AbstractLattice{T, 3}
    a::Float64
    data::A
end
FCCLattice(L::Integer) = FCCLattice(1.0, fill(0, sitesperunitcell(FCCLattice, L)))

function coord(L::FCCLattice, I::Index{3})::Point3f0
    a = L.a
    ix,iy,iz = Tuple(I) .- 1
    z = a/2*iz
    x = a/2*ix
    if iseven(iz)
        y = ifelse(iseven(ix), a*iy, a*iy + a/2)
    else
        y = ifelse(isodd(ix), a*iy, a*iy + a/2)
    end
    return Point3f0(x,y,z)
end

function index(L::FCCLattice, p)
    x,y,z = round.(2/L.a*p)
    ix = round(Int, x) + 1
    iz = round(Int, z) + 1
    if isodd(iz)
        iy = ifelse(isodd(ix), y/2+1, (y-1)/2+1)
    else
        iy = ifelse(iseven(ix), y/2+1, (y-1)/2+1)
    end
    return CartesianIndex(ix, round(Int, iy), iz)
end

Base.@propagate_inbounds function neighbors!(nn::Neighbors{12,3}, L::FCCLattice, I)
    l,m,n = Tuple(I)
    if isodd(n)
        mnew = m + ifelse(isodd(l), -1, +1)
    else
        mnew = m + ifelse(iseven(l), -1, +1)
    end
    nn[1] = CartesianIndex(l-1, m, n)
    nn[2] = CartesianIndex(l+1, m, n)
    nn[3] = CartesianIndex(l+1, mnew, n)
    nn[4] = CartesianIndex(l-1, mnew, n)
    # plane above
    nn[5] = CartesianIndex(l-1, m, n+1)
    nn[6] = CartesianIndex(l, mnew, n+1)
    nn[7] = CartesianIndex(l+1, m, n+1)
    nn[8] = CartesianIndex(l, m, n+1)
    # plane below
    nn[9] = CartesianIndex(l-1, m, n-1)
    nn[10] = CartesianIndex(l, mnew, n-1)
    nn[11] = CartesianIndex(l+1, m, n-1)
    nn[12] = CartesianIndex(l, m, n-1)
    nothing
end

function nneighbors(::Type{FCCLattice{T, A}}, N, I) where {T, A}
    coord = coordination(FCCLattice)

end

## --- END FCC lattice (3D) -- ##

coord(L::AbstractLattice{<:Any, N}, v::AbstractArray{<:Index{N}}) where {N} = coord.(Ref(L), v)

dimension(::AbstractLattice{T, N}) where {T,N} = N
dimension(::Type{LT}) where LT<:AbstractLattice{T, N} where {T,N} = N
dimension(::Type{LineLattice}) = 1
dimension(::Type{HexagonalLattice}) = 2
dimension(::Type{CubicLattice}) = 3
dimension(::Type{FCCLattice}) = 3
dimension(::Type{HCPLattice}) = 3

coordination(L::RealLattice) = coordination(typeof(L))
coordination(::Type{LineLattice{T, A}}) where {T, A} = 2
coordination(::Type{HexagonalLattice{T, A}}) where {T, A} = 6
# coordination(::Type{CubicLattice{T, A}}) where {T, A} = 6
coordination(::Type{T}) where T<:CubicLattice = 6
coordination(::Type{T}) where T<:FCCLattice = 12
coordination(::Type{HCPLattice{T, A}}) where {T, A} = 12

spacings(L::LineLattice) = (L.a,)
spacings(L::HexagonalLattice) = (L.a, L.a)
spacings(L::CubicLattice) = (L.a, L.a, L.a)
spacings(L::FCCLattice) = (L.a, L.a, L.a)

sitesperunitcell(::Type{FCCLattice}, L) = (2L+1, L+1, 2L+1)
sitesperunitcell(::Type{LT}, L) where LT<:AbstractLattice = ntuple(_->L+1, dimension(LT))

"""
    density(L::RealLattice, I)

Calculates `(occupied sites)/(no. of neighbors)`

"Occupied" means != zero(eltype(L.data))
"""
density(L::RealLattice, I) = density!(Neighbors(L), L, I)
function density!(nn::Neighbors, L::RealLattice{G}, I) where G
    tot = nneighbors(L, I)
    neighbors!(nn, L, I)
    nz =  count(x->!out_of_bounds(x, size(L)) && L.data[x]!=zero(G), nn) ## TODO: makes assumption about the numerical value that repr. unoccupied
    return nz/tot
end

###################
## Intersections ##
###################

function conicsection(L::AbstractLattice{<:Any, 3}, coords, Ω; axis=Point3f0(0,0,-1), o=Lattices.coord(L, Lattices.midpoint(L)))
    ## !! WARNING: ϕ is the azimuth angle in CoordinateTransformations !!
    cts = SphericalFromCartesian()
    # rotY = @SMatrix [ cos(ϕoff) 0 -sin(ϕoff);
    #                  0         1    0;
    #                  sin(ϕoff) 0  cos(ϕoff) ]

    # rotZ = @SMatrix [ cos(θoff) -sin(θoff) 0;
    #                   sin(θoff)  cos(θoff) 0;
    #                   0         0          1]
    z = Point3f0(0,0,1)
    _axis = cross(axis, z)
    # @show _axis
    if iszero(_axis)
        _axis = Point3f0(0,1,0)
    else
        _axis = normalize(_axis)
    end
    angle_zdir = acos(dot(z, axis)/norm(axis))
    # @show angle_zdir
    rot = AngleAxis(angle_zdir, _axis...)
    # @show rot
    filter(coords) do p
      q = cts(rot*(p-o))
      q.ϕ+π/2 - eps(Float32) ≤ acos(1-Ω/(2π))
    end
end

## END MODULE ##
end
