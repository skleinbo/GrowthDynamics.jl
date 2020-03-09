module TumorConfigurations

import OpenCLPicker: @opencl
import ..Lattices
import GeometryTypes: Point2f0, Point3f0
@opencl import .OffLattice: FreeSpace,distMatrix
import LightGraphs: DiGraph, add_vertex!, add_edge!
import Base: push!, show
import StatsBase

export
    TumorConfiguration,
    MetaData,
    push!,
    nolattice_state,
    single_center2,
    single_center3,
    single_center3_cubic,
    uniform_line,
    uniform_circle,
    uniform_circle_free,
    uniform_sphere

##-- METADATA for efficiently storing population information --##
const MetaDatum{T} = Tuple{T, Int64, Float64, Vector{Int64}, Tuple{Int64,Float64}}
mutable struct MetaData{T}
    genotypes::Vector{T}
    npops::Vector{Int64}
    fitnesses::Vector{Float64}
    snps::Vector{Vector{Int64}}
    ages::Vector{Tuple{Int64, Float64}}  ## (simulation t, real t) when a genotype entered.
end
MetaData(T::DataType) = MetaData(T[], Int64[], Float64[], Vector{Int64}[], Tuple{Int64,Float64}[])
MetaData(a::Tuple{T, Int64, Float64, Vector{Int64}}) where {T} = MetaData{T}([a[1]],[a[2]],[a[3]],[a[4]])
Base.getindex(M::MetaData{T}, i) where {T} = MetaData{T}(M.genotypes[i], M.npops[i], M.fitnesses[i], M.snps[i], M.ages[i])
Base.getindex(M::MetaData{T}, i::Integer) where {T} =
    (genotype = M.genotypes[i], npop = M.npops[i], fitness = M.fitnesses[i], snps = M.snps[i], age = M.ages[i])
Base.lastindex(M::MetaData{T}) where {T} = length(M.genotypes)
function Base.setindex!(M::MetaData{T}, D::Tuple{T, Int64, Float64, Vector{Int64}, Tuple{Int64,Float64}}, i) where {T}
    M.genotypes[i] = D[1]
    M.npops[i] = D[2]
    M.fitnesses[i] = D[3]
    M.snps[i] = D[4]
    M.ages[i] = D[5]
    D
end

const DEFAULT_META_DATA = (1, 1.0, Int64[], (0,0.0))

##--                                                        --##

mutable struct TumorConfiguration{T<:Lattices.AbstractLattice}
    lattice::T
    Phylogeny::DiGraph
    meta::MetaData
    t::Int
    treal::Float64
    observables::Dict{Symbol, Any}
end
function TumorConfiguration(lattice::Lattices.AnyTypedLattice{T}, Phylogeny::DiGraph) where {T}
    TumorConfiguration(lattice, Phylogeny, MetaData(T), 0, 0.0, Dict{Symbol, Any}())
end
Base.getindex(T::TumorConfiguration,ind...) = T.lattice.data[ind...]
Base.getindex(T::TumorConfiguration) = T.lattice.data

function Base.setindex!(T::TumorConfiguration,v,ind...)
    L = T.lattice.data
    g_old = L[ind...]
    if L[ind...] == v
        return v
    end
    if g_old != 0
        g_id = findfirst(x->x==g_old, T.meta.genotypes)
        T.meta.npops[g_id] -= 1
    end
    if !(v in T.meta.genotypes)
        push!(T, v)
        T.meta.npops[end] = 1
    else
        g_id = findfirst(x->x==v, T.meta.genotypes)
        T.meta.npops[g_id] += 1
    end
    L[ind...] = v
    v
end

"""
    nolattice_state(N)

Unstructered model with carrying capacity `N`, one genotype and one individual with fitness 1.0.
"""
nolattice_state(N::Int) = begin
    state = TumorConfiguration(Lattices.NoLattice(N), DiGraph())
    push!(state, 1)
    state.meta.npops[end] = 1
    state.meta.fitnesses[end] = 1.0
    state.meta.ages[end] = (0, 0.0)
    state
end

"Add a new _unconnected_ genotype to a TumorConfiguration"
function Base.push!(S::TumorConfiguration{<:Lattices.AnyTypedLattice{T}}, g::T) where {T}
    add_vertex!(S.Phylogeny)
    push!(S.meta.genotypes, g)
    push!(S.meta.npops, 0)
    push!(S.meta.fitnesses, 1.0)
    push!(S.meta.snps, Int64[])
    push!(S.meta.ages, (S.t, S.treal))
    nothing
end

function Base.push!(S::TumorConfiguration{<:Lattices.AnyTypedLattice{T}}, M::MetaDatum{T}) where {T}
    add_vertex!(S.Phylogeny)
    push!(S.meta.genotypes, M[1])
    push!(S.meta.npops, M[2])
    push!(S.meta.fitnesses, M[3])
    push!(S.meta.snps, M[4])
    push!(S.meta.ages, M[5])
    nothing
end


"""
    uniform_line(L [, g=0])

One-dimension system filled with genotype `g`.
"""
function uniform_line(L, g=0)
    G = DiGraph()
    lattice = Lattices.LineLattice(L, 1.0, fill(g, L))
    state = TumorConfiguration(lattice, G)
    if g!=0
        push!(state, (g, L, 1.0, Int64[], (0,0.0)) )
    end
    state
end

"""
    single_center2(N [;g1=1,g2=2])

Initialize a single cell of genotype `g2` at the midpoint of hexagonal lattice filled with `g1`.
"""
function single_center2(N::Int;g1=1,g2=2)
    G = DiGraph()
    lattice = Lattices.HexagonalLattice(N,N,1.0,fill(g1,N,N))
    state = TumorConfiguration(lattice, G)
    midpoint = CartesianIndex(div(N,2),div(N,2))

    state[midpoint] = g2

    counts = StatsBase.countmap(reshape(lattice.data,N^2))
    if g1!=0
        push!(state, (g1, counts[g1], 1.0, Int64[]))
    end
    if g2!=0
        push!(state, (g2, counts[g2], 1.0, Int64[]))
    end
    if g1!=0 && g2!=0
         add_edge!(G, 2, 1)
    end

    return state
end
"""
    single_center3(N [;g1=1,g2=2])

Initialize a single cell of genotype `g2` at the midpoint of HCP lattice filled with `g1`.
"""
function single_center3(N::Int;g1=1,g2=2)
    G = DiGraph()
    lattice = Lattices.HCPLattice(N,N,N,1.0,fill(g1,N,N,N))
    state = TumorConfiguration(lattice, G)
    midpoint = CartesianIndex(div(N,2),div(N,2),div(N,2))

    state[midpoint] = g2

    counts = StatsBase.countmap(reshape(lattice.data,N^3))
    if g1!=0
        push!(state, (g1, DEFAULT_META_DATA...) )
        state.meta.npops[1] = counts[g1]
    end
    if g2!=0
        push!(state, (g2,  DEFAULT_META_DATA...) )
    end
    if g1!=0 && g2!=0
         add_edge!(G, 2, 1)
    end

    return state
end

"""
    single_center3_cubic(N [;g1=1,g2=2])

Initialize a single cell of genotype `g2` at the midpoint of a cubic lattice filled with `g1`.
"""
function single_center3_cubic(N::Int;g1=1,g2=2)
    G = DiGraph()
    lattice = Lattices.CubicLattice(N,N,N,1.0,fill(g1,N,N,N))
    state = TumorConfiguration(lattice, G)
    midpoint = CartesianIndex(div(N,2),div(N,2),div(N,2))

    state[midpoint] = g2

    counts = StatsBase.countmap(reshape(lattice.data,N^3))
    if g1!=0
        push!(state, (g1, DEFAULT_META_DATA...) )
        state.meta.npops[1] = counts[g1]
    end
    if g2!=0
        push!(state, (g2,  DEFAULT_META_DATA...) )
    end
    if g1!=0 && g2!=0
         add_edge!(G, 2, 1)
    end

    return state
end

function uniform_sphere(N::Int,f=1/10,g1=1,g2=2)::Lattices.HCPLattice{Int}
    state = Lattices.HCPLattice(N,N,N,1.0,fill(g1,N,N,N))
    mid = [div(N,2),div(N,2),div(N,2)]

    function fill_neighbors!(state,m,n,l)
        nn = Lattices.neighbors(state, m,n,l)
        for neigh in nn
            if Lattices.out_of_bounds(neigh...,state.Na) continue end
            state.data[CartesianIndex(neigh)] = g2
            if count(x->x==g2, state.data)/N^3 >= f
                return nn
            end
        end
        return nn
    end

    state.data[mid...] = g2
    while count(x->x==g2, state.data)/N^3 < f
        for nn in findall(x->x==g2,state.data)
            # println(nn)
            fill_neighbors!(state,nn)
        end
    end

    return state
end


@opencl function uniform_circle_free(N::Int, Nmax::Int, f=1/10,g1=1,g2=2,d=0f0)
    r = sqrt(f/pi)
    midpoint = Point2f0(0.5,0.5)

    positions = rand(Float32,2,Nmax)
    dMat = distMatrix(positions)
    genotypes = zeros(Int32, Nmax)
    birthrates = zeros(Float32, Nmax)
    deathrates = fill(Float32(d), Nmax)

    _mask = zeros(UInt32,Nmax)
    _mask[1:N] .= UInt32(1)

    G = TwoPhylogeny()

    for n in 1:N
        if norm(positions[:,n]-midpoint) < r
            genotypes[n] = g2
        else
            genotypes[n] = g1
        end
    end
    FreeSpace{eltype(genotypes)}(Nmax, G, positions,genotypes,birthrates,deathrates,dMat,_mask)
end

"""
    uniform_circle(L [,f=1/10,g1=1,g2=2])

Hexagonal lattice state filled with `g1`.
Disk of filling fraction `f` with genotype `g2` at the center.
"""
function uniform_circle(N::Int,f=1/10,g1=1,g2=2)::TumorConfiguration{Lattices.HexagonalLattice{Int}}
    G = DiGraph()
    lattice = Lattices.HexagonalLattice(N,N,1.0,fill(g1,N,N))
    state = TumorConfiguration(lattice, G)


    if f==0.
        return state
    end

    mid = CartesianIndex(div(N,2),div(N,2))

    function fill_neighbors!(lattice,ind::CartesianIndex)
        m,n = Tuple(ind)
        nn = Lattices.neighbors(lattice, ind)
        for neigh in nn
            if Lattices.out_of_bounds(neigh,lattice.Na) continue end
            lattice.data[neigh] = g2
            if count(x->x==g2, lattice.data)/N^2 >= f
                return nn
            end
        end
        return nn
    end

    lattice.data[mid] = g2
    while count(x->x==g2, lattice.data)/N^2 < f
        for nn in findall(x->x==g2,lattice.data)
            # println(nn)
            fill_neighbors!(lattice,nn)
        end
    end

    counts = StatsBase.countmap(reshape(lattice.data,N^2))
    if g1!=0
        push!(state, (g1, counts[g1], 1.0, Int64[], (0,0.0)))
    end
    if g2!=0
        push!(state, (g2, counts[g2], 1.0, Int64[], (0,0.0)))
    end
    if g1!=0 && g2!=0
         add_edge!(G, 2, 1)
    end

    return state
end

##--END module--
end
