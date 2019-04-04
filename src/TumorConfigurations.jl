module TumorConfigurations

import OpenCLPicker: @opencl
import ..Lattices
import GeometryTypes: Point2f0, Point3f0
@opencl import .OffLattice: FreeSpace,distMatrix
import LightGraphs: DiGraph
import MetaGraphs: MetaDiGraph, add_vertex!, add_edge!, set_prop!, set_indexing_prop!

export
    nolattice_state,
    single_center,
    single_center3,
    random_center,
    uniform_line,
    uniform_square,
    uniform_circle,
    uniform_circle2,
    uniform_circle_free,
    uniform_sphere,
    uniform_sphere2,
    biallelic_layered,
    biallelic_cornered,
    TumorConfiguration

mutable struct TumorConfiguration{T<:Lattices.AbstractLattice}
    lattice::T
    Phylogeny::MetaDiGraph
    t::Int
    treal::Float64
end
TumorConfiguration(lattice::T, Phylogeny::MetaDiGraph) where T<:Lattices.AbstractLattice = TumorConfiguration(lattice, Phylogeny, 0, 0.0)
Base.getindex(T::TumorConfiguration,ind...) = T.lattice.data[ind...]
Base.getindex(T::TumorConfiguration) = T.lattice.data
Base.setindex!(T::TumorConfiguration,v,ind...) = T.lattice.data[ind...] = v

nolattice_state(N::Int) = TumorConfiguration(Lattices.NoLattice(N), OnePhylogeny())


"Returns a DiGraph with one vertex and {T=>0,genotype=>g} attribute."
function OnePhylogeny(g=1)
    G = MetaDiGraph(1)
    set_indexing_prop!(G,1,:genotype,g)
    set_prop!(G,1,:T,0)
    set_prop!(G,1,:npop,0)
    G
end

"Returns a DiGraph with edge 2=>1 and {T=>0,genotype=>1,2} attribute on both verticies."
function TwoPhylogeny(gv=[1,2])
    G = MetaDiGraph(2)
    add_edge!(G,(2,1))
    for v in [1,2]
        set_indexing_prop!(G,v,:genotype,gv[v])
        set_prop!(G,v,:T,0)
        set_prop!(G,v,:npop,0)
    end
    G
end

"""
Initialize a single cell of genotype `1` at the midpoint of an empty lattice.
"""
function single_center(N::Int;g1=1,g2=2)::Lattices.HexagonalLattice{Int}
    midpoint = [div(N,2),div(N,2)]
    state = fill(g1,N,N)
    state[midpoint] = g2
    return Lattices.HexagonalLattice(N,N,1.0,state)
end

"""
Bla
"""
function single_center3(N::Int;g1=1,g2=2)::Lattices.HCPLattice{Int}
    midpoint = [div(N,2),div(N,2), div(N,2)]
    state = fill(g1,N,N,N)
    state[midpoint] = g2
    return Lattices.HCPLattice(N,N,N,1.0,state)
end

"""
Seed a `LxL` center with random genotypes.
"""
function random_center(N::Int,L)::Array{Int64,2}
    midpoint = (div(N,2),div(N,2))
    state = zeros(Int64,N,N)
    state[ div(N,2)-(L-1):div(N,2)+L, div(N,2)-(L-1):div(N,2)+L ] = rand(1:100,2*L,2*L)
    return state
end

"""
Fill center square with genotype `g1` with fraction `f1`, and gt `g2` with fraction `f2`.
"""
function uniform_square(N::Int,f1=1.0,f2=0.1,g1=1,g2=2)::Array{Int64,2}
    midpoint = [div(N,2),div(N,2)]
    state = zeros(Int64,N,N)

    a = round(Int,max(1,div(N,2)*(1-sqrt(f1))))
    b = round(Int,div(N,2)*(1+sqrt(f1)))
    state[ a:b, a:b ] = g1

    a = round(Int,max(1,div(N,2)*(1-sqrt(f2))))
    b = round(Int,div(N,2)*(1+sqrt(f2)))
    state[ a:b, a:b ] = g2

    return state
end

"""
Fill line of fraction `f1` with gt `g2`; rest is gt `g1`.
"""
function uniform_line(N::Int,f=1/10,g1=1,g2=2)::Lattices.LineLattice{Int}
    state = fill(g1,N)
    mid = div(N,2)
    state[mid-round(Int,N/2*f)+1:mid+round(Int,N/2*f)] = g2
    return Lattices.LineLattice(N,1.0,state)
end


"""
Fill sphere of fraction `f1` with gt `g2`; rest is gt `g1`.
"""
function uniform_sphere(N::Int,f=1/10,g1=1,g2=2)::Lattices.HCPLattice{Int}
    state = fill(g1,N,N,N)
    mid = [div(N,2),div(N,2),div(N,2)]
    for m in 1:N, n in 1:N, l in 1:N
        if (m-mid[1])^2+(n-mid[2])^2+(l-mid[3])^2 <= N^2*(3*f/(4pi))^(2/3)
            state[m,n,l] = g2
        end
    end
    return Lattices.HCPLattice(N,N,N,1.0,state)
end

function uniform_sphere2(N::Int,f=1/10,g1=1,g2=2)::Lattices.HCPLattice{Int}
    state = Lattices.HCPLattice(N,N,N,1.0,fill(g1,N,N,N))
    mid = [div(N,2),div(N,2),div(N,2)]

    function fill_neighbours!(state,m,n,l)
        nn = Lattices.neighbours(state, m,n,l)
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
            fill_neighbours!(state,nn)
        end
    end

    return state
end



function uniform_circle(N::Int,f=1/10,g1=1,g2=2)::Lattices.HexagonalLattice{Int}
    state = fill(g1,N,N)
    if N^2*f < 3/4
        return state
    end
    mid = [div(N,2),div(N,2)]
    for m in 1:N, n in 1:N
        if (m-mid[1])^2+(n-mid[2])^2 <= ( sqrt(N^2*f/3-1/4)-1/2 )^2
            state[m,n] = g2
        end
    end
    return Lattices.HexagonalLattice(N,N,1.0,state)
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

function uniform_circle2(N::Int,f=1/10,g1=1,g2=2)::TumorConfiguration{Lattices.HexagonalLattice{Int}}
    if g1==0
        G = OnePhylogeny(g2)
    elseif g2==0
        G = OnePhylogeny(g1)
    else
        G = TwoPhylogeny([g1,g2])
    end
    state = Lattices.HexagonalLattice(N,N,1.0,fill(g1,N,N))

    if f==0.
        return state
    end

    mid = CartesianIndex(div(N,2),div(N,2))

    function fill_neighbours!(state,ind::CartesianIndex)
        m,n = Tuple(ind)
        nn = Lattices.neighbours(state, ind)
        for neigh in nn
            if Lattices.out_of_bounds(neigh,state.Na) continue end
            state.data[neigh] = g2
            if count(x->x==g2, state.data)/N^2 >= f
                return nn
            end
        end
        return nn
    end

    state.data[mid] = g2
    while count(x->x==g2, state.data)/N^2 < f
        for nn in findall(x->x==g2,state.data)
            # println(nn)
            fill_neighbours!(state,nn)
        end
    end

    return TumorConfiguration(state, G)
end


function biallelic_layered(N,f,g1=1,g2=2)::Array{Int64,2}
    state = fill(g1,N,N)
    state[1:round(Int,N*f),1:N] = g2
    return state
end

function biallelic_cornered(N,f,g1=1,g2=2)::Array{Int64,2}
    state = fill(g1,N,N)
    state[1:round(Int,N*sqrt(f)),1:round(Int,N*sqrt(f))] = g2
    return state
end
##--END module--
end
