module TumorConfigurations

import Base: axes, copyto!, copy, getindex, length, push!, resize!, setindex!, similar, show, zero, @propagate_inbounds
import Base.Iterators: product
import LinearAlgebra: norm
import LightGraphs: SimpleDiGraph, add_vertex!, add_vertices!, add_edge!, nv
import ..Lattices
import ..Lattices: coord, dimension, index, isonshell, radius, realsize, midpoint, dist, spacings
import GeometryBasics: Point2f0, Point3f0
import StatsBase
import CoordinateTransformations: SphericalFromCartesian

export TumorConfiguration,
    MetaData,
    push!,
    nolattice_state,
    single_center,
    uniform,
    half_space,
    spheref, spherer,
    sphere_with_diverse_outer_shell,
    sphere_with_single_mutant_on_outer_shell

##-- METADATA for efficiently storing population information --##
const MetaDatumFields = (:genotype, :npop, :fitness, :snps, :age)
const MetaDatumFieldTypes{T} = Tuple{T,Int64,Float64,Vector{Int64},Tuple{Int64,Float64}}
const MetaDatum{T} = NamedTuple{MetaDatumFields,MetaDatumFieldTypes{T}}
const DEFAULT_META_DATA = (0, 1.0, Int64[], (0, 0.0))
"""
    MetaDatum

NamedTuple to store information about one genotype.
"""
function MetaDatum(A::MetaDatumFieldTypes)
    NamedTuple{MetaDatumFields}(A)
end

function MetaDatum(g)
    MetaDatum((g, DEFAULT_META_DATA...))
end

mutable struct MetaData{T}
    _len::Int64
    genotypes::Vector{T}
    npops::Vector{Int64}
    fitnesses::Vector{Float64}
    snps::Vector{Vector{Int64}}
    ages::Vector{Tuple{Int64,Float64}}  ## (simulation t, real t) when a genotype entered.
    misc::Dict{Any,Any} # store anything else in here.
end

"""
    MetaData(T::DataType)

Empty MetaData for genotype-type T.
"""
MetaData(T::DataType) = MetaData(0, T[], Int64[], Float64[], Vector{Int64}[], Tuple{Int64,Float64}[], Dict())

function MetaData{T}(::UndefInitializer, n) where T
    return MetaData(0,
        Vector{T}(undef, n),
        Vector{Int64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Vector{Int64}}(undef, n),
        Vector{Tuple{Int64,Float64}}(undef, n),
        Dict()
    )
end

"""
    MetaData(M::Union{MetaDatum, Tuple})

Construct MetaData from single datum. Argument can be an appropriate tuple or named tuple.
"""
MetaData(M::MetaDatum) = MetaData(values(M))
MetaData(a::MetaDatumFieldTypes{T}) where {T} = MetaData{T}(1, [a[1]], [a[2]], [a[3]], [a[4]], [a[5]], Dict())
"""
    MetaData(g::Vector{T}, n::Vector{<:Integer})

Construct MetaData from vectors of genotypes and population sizes.
* Fitnesses default to 1.0
* SNPs default to empty.
* Ages default to (0, 0.0)
* misc defaults to an empty dictionary.
"""
function MetaData(g::Vector{T}, n::Vector{<:Integer}) where {T}
    N = length(n)
    if length(g) != N
        throw(ArgumentError("Lengths of arguments do not match."))
    end
    fitnesses = fill(1.0, N)
    snps = fill(Int64[], N)
    ages = fill((0, 0.0), N)
    MetaData(N, g, n, fitnesses, snps, ages, Dict())
end

length(M::MetaData) = M._len

function Base.similar(M::MetaData{T}) where T
    return similar(M, M._len)
end
function Base.similar(::MetaData{T}, len) where T
    return MetaData{T}(undef, len)
end

@propagate_inbounds function Base.copyto!(dest::MetaData{T}, src::MetaData{T}) where T
    @boundscheck if length(src) > length(dest.genotypes)
        throw(BoundsError())
    end
    dest._len = src._len
    dest.misc = copy(src.misc)
    for field in setdiff(fieldnames(src), [:_len, :misc])
        copyto!(getproperty(dest, field), getproperty(dest, field))
    end
    return dest
end

@inline function _pluralize(field::Symbol)
    if field == :genotype
        return :genotypes
    elseif field == :npop
        return :npops
    elseif field == :fitness
        return :fitnesses
    elseif field == :snps
        return :snps
    elseif field == :age
        return :ages
    else
        throw(ArgumentError("Unkown field $field"))
    end    
end

@propagate_inbounds function gindex(M::MetaData{T}, g::T) where T
    id = findfirst(x -> x == g, @view M.genotypes[begin:length(M)])
    @boundscheck if id === nothing
        throw(ArgumentError("Unknown genotype $g."))
    end
    id
end

@propagate_inbounds function Base.getindex(M::MetaData{T}; g) where {T}
    M[gindex(M, g)]
end

@propagate_inbounds function Base.getindex(M::MetaData{T}, i::Integer) where {T}
    @boundscheck if i>M._len
        throw(BoundsError(M, i))
    end
    (genotype = M.genotypes[i], npop = M.npops[i], fitness = M.fitnesses[i], snps = M.snps[i], age = M.ages[i])
end
@propagate_inbounds function Base.getindex(M::MetaData{T}, ::Colon) where {T}
    I = 1:M._len
    @inbounds M[I]
end

@propagate_inbounds function Base.getindex(M::MetaData{T}, I) where {T}
    @boundscheck begin
        within = intersect(I, 1:M._len)==I
        if !within
            throw(BoundsError())
        end
    end
    N = MetaData{T}(
        0,
        M.genotypes[I],
        M.npops[I],
        M.fitnesses[I],
        M.snps[I],
        M.ages[I],
        M.misc
    )
    N._len = length(N.genotypes)
    return N
end

@propagate_inbounds function getindex(M::MetaData, i::Integer, field::Symbol)
    mfield = _pluralize(field)
    getindex(getproperty(M, mfield), i)
end

@propagate_inbounds function getindex(M::MetaData{T}, field::Symbol; g::T) where T
    M[gindex(M, g), field]
end

@propagate_inbounds function getindex(M::MetaData{T}, ::Colon, field::Symbol) where T
    return getfield(M, field)[begin:M._len]
end

@inline @propagate_inbounds function getindex(M::MetaData{T}, I, field::Symbol) where T
    @boundscheck begin
        within = intersect(I, 1:M._len)==I
        if !within
            throw(BoundsError())
        end
    end
    return getfield(M, field)[I]
end

function resize!(M::MetaData, n::Integer)
    if n<length(M)
        throw(BoundsError("Requested size is less than current size."))
    end
    if n==M._len
        return M
    end
    _resize!(M, n)
    M._len = n
    M
end

function _resize!(M::MetaData, n::Integer)
    if length(M.genotypes) == n
        return M
    end
    for field in fieldnames(MetaData)
        if field in [:_len, :misc]
            continue
        else
            resize!(getproperty(M, field), n)
        end
    end
    # Mnew = similar(M, n)
    # copyto!(Mnew, M)
    M
end    


Base.lastindex(M::MetaData{T}) where {T} = length(M)

Base.push!(M::MetaData{T}, g::T) where T = push!(M, MetaDatum(g))
Base.push!(M::MetaData, D::MetaDatum) = push!(M, values(D))
@inline @propagate_inbounds function Base.push!(M::MetaData{T}, D::MetaDatumFieldTypes{T}) where T
    # @boundscheck if D[1] == zero(T) || D[1] in @view M.genotypes[begin:M._len]
    #     throw(ArgumentError("Invalid genotype $(D[1]): either already present or zero."))
    # end
    i = lastindex(M)+1
    int_length = length(M.genotypes) # internal length
    if i >= int_length
        _resize!(M, ceil(Int64, max(1, int_length*2.0)))
    end
    M._len = i
    setindex!(M, D, i)
end

@propagate_inbounds Base.setindex!(M::MetaData, D::MetaDatum, i::Integer) = setindex!(M, values(D), i)

@propagate_inbounds @inline function Base.setindex!(M::MetaData{T}, D::MetaDatumFieldTypes{T}, i::Integer) where {T}
    @boundscheck if i>lastindex(M)
        throw(BoundsError(M, i))
    end
    M.genotypes[i] = D[1]
    M.npops[i] = D[2]
    M.fitnesses[i] = D[3]
    M.snps[i] = D[4]
    M.ages[i] = D[5]
    D
end

@propagate_inbounds function setindex!(M::MetaData, v, i::Integer, field::Symbol)
    @boundscheck if i > length(M)
        throw(BoundsError(M,i))
    end
    mfield = _pluralize(field)
    setindex!(getproperty(M, mfield), v, i)
    return v
end

@propagate_inbounds function setindex!(M::MetaData, v, field::Symbol; g)
    i = gindex(M, g)
    setindex!(M, v, i, field)
end

@propagate_inbounds Base.axes(M::MetaData) = (axes(M, 1),)
@propagate_inbounds function Base.axes(M::MetaData, d::Integer)
    if d==1
        return Base.OneTo(lastindex(M))
    else
        return Base.OneTo(1)
    end
end


##--                                                        --##

## Certain values on the lattice are special. 
## For example, we need a way to identify the empty site.
## We use `zero` for that. If zero is undefined for the 
## type you are using, define it, e.g
zero(::Type{String}) = "0"

mutable struct TumorConfiguration{T <: Lattices.AbstractLattice}
    lattice::T
    phylogeny::SimpleDiGraph
    meta::MetaData
    t::Int
    treal::Float64
    observables::Dict{Symbol, Any}
end

"""
    TumorConfiguration(lattice, [phylogeny])

Wraps an existing lattice in a TumorConfiguration. Calculates meta.npops automatically.

If `phylogeny` is not given, it defaults to an empty graph.
"""
function TumorConfiguration(lattice::Lattices.RealLattice{T}, phylogeny::SimpleDiGraph = SimpleDiGraph()) where {T}
    counts_dict = StatsBase.countmap(lattice.data, alg = :dict)
    if haskey(counts_dict, zero(T))
        delete!(counts_dict, zero(T))
    end
    genotypes = collect(keys(counts_dict))
    npops = collect(values(counts_dict))
    metadata = MetaData(genotypes, npops)
    add_vertices!(phylogeny, length(genotypes))
    TumorConfiguration(lattice, phylogeny, metadata, 0, 0.0, Dict{Symbol,Any}())
end
function TumorConfiguration(nolattice::Lattices.NoLattice{T}) where T
    metadata = MetaData(T)
    phylogeny = SimpleDiGraph()
    TumorConfiguration(nolattice, phylogeny, metadata, 0, 0.0, Dict{Symbol,Any}())
end

@propagate_inbounds Base.getindex(T::TumorConfiguration, ind...) = getindex(T.lattice.data, ind...)

@propagate_inbounds Base.setindex!(T::TumorConfiguration, v, ind::CartesianIndex) = setindex!(T, v, Tuple(ind)...)

@propagate_inbounds function Base.setindex!(T::TumorConfiguration, v, ind::Vararg{Int64})
    z = zero(eltype(T.lattice.data))
    L = T.lattice.data
    g_old = L[ind...]
    if L[ind...] == v
        return v
    end
    if v != z
        if !(v in @view T.meta.genotypes[begin:T.meta._len])
            push!(T, v)
            T.meta[end, :npop] = 1
        else
            T.meta[g=v, :npop] += 1
        end
    end
    if g_old != z
        T.meta[g=g_old, :npop] -= 1
    end
    L[ind...] = v
    v
end

"Add a new _unconnected_ genotype to a TumorConfiguration."
@inline @propagate_inbounds function Base.push!(S::TumorConfiguration{<:Lattices.TypedLattice{T}}, g::T) where {T}
    push!(S.meta, (g, 0, 1.0, Int64[], (S.t, S.treal)))
    add_vertex!(S.phylogeny)
    lastindex(S.meta)
end

function Base.push!(S::TumorConfiguration{<:Lattices.TypedLattice{T}}, M::MetaDatum{T}) where {T}
    push!(S.meta, M)
    add_vertex!(S.phylogeny)
    lastindex(S.meta)
end

#####################################################################
## -- Convenience constructors for different initial geometries -- ##
#####################################################################

"""
    nolattice_state()

Unstructered model.
One genotype 1::Int64 with one individual with fitness 1.0.
"""
nolattice_state() = begin
    state = TumorConfiguration(Lattices.NoLattice())
    push!(state, 1)
    state.meta[end, :npop] = 1
    state.meta[end, :fitness] = 1.0
    state
end


"""
    uniform(::Type{LT<:RealLattice}, L; g=0, a=1.0)

Return system on a lattice of type `LT` with linear extension `L`, filled with genotype `g`.

# Example

    uniform(HexagonalLattice, 128; g=1)
"""
function uniform(T::Type{LT}, L::Int; g=0) where LT<:Lattices.RealLattice
    dim = Lattices.dimension(T)
    lattice = LT(1.0, fill(g, fill(L, dim)...))
    state = TumorConfiguration(lattice)

    return state
end

"""
    single_center(::Type{RealLattice}, L; g1=1,g2=2)

Initialize a single cell of genotype `g2` at the midpoint of the given lattice type, filled with `g1`.
"""
function single_center(::Type{LT}, L::Int; g1=0, g2=1) where LT<:Lattices.RealLattice
    state = uniform(LT, L; g=g1)
    mid = midpoint(state.lattice)
    state[mid] = g2

    return state
end

function half_space(::Type{LT}, L::Int; f=1/2, g1=0, g2=1) where LT<:Lattices.RealLattice
    if !(0.0<=f<=1)
        throw(ArgumentError("filling fraction must be between 0 and 1."))
    end
    state = uniform(LT, L; g=g1)
    fill_to = round(Int, f*size(state.lattice)[end])
    for ind in product(axes(state.lattice.data)[1:end-1]..., 1:fill_to)
        state[ind...] = g2
    end
    state
end
function spherer(::Type{LT}, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice
    if !(0.0<=r)
        throw(ArgumentError("radius must be positive."))
    end
    state = uniform(LT, L; g=g1)
    if g2==g1 || r==0.0
        return state
    end

    mid = coord(state.lattice, midpoint(state.lattice))

    all_indices = CartesianIndices(state.lattice.data)
    # dist_matrix = map(x->dist(state.lattice, x, mid)<=r, all_indices)

    for I in eachindex(all_indices)
        p = coord(state.lattice, I)
        if norm(p-mid)<=r #|| isonshell(state.lattice, p, r, mid)
            state[I] = g2
        end
    end
    if g1!=0 && g2!=0
        add_edge!(state.phylogeny, g2, g1)
    end
    state
end

function spheref(::Type{LT}, L::Int; f = 1 / 10, g1=1, g2=2) where LT<:Lattices.RealLattice
    if !(0.0<=f<=1)
        throw(ArgumentError("f must be between 0 and 1."))
    end
    state = uniform(LT, L; g=g1)
    if g2==g1 || f==0.0
        return state
    end
    N = length(state.lattice)
    a = spacings(state.lattice)[1]
    r_old = radius(f*N*a, dimension(state.lattice)) # start with radius implied by the real-space volume Nfa
    r_new = r_old

    mid = midpoint(state.lattice)

    all_indices = CartesianIndices(state.lattice.data)
    dist_matrix = map(x->dist(state.lattice, x, mid), all_indices)

    n, n_old = count(I->I < r_new, dist_matrix), 0
    while n/length(state.lattice) < f
        r_old = r_new
        r_new += a
        n_old = n
        n = count(I->I < r_new, dist_matrix)
        if n-n_old == 0
            break
        end
    end
    sphere = findall(I->I < r_new, dist_matrix)
    for ind in sphere
        state[ind] = g2
    end
    state, sphere
end

function sphere_with_diverse_outer_shell(::Type{LT}, L::Int; r) where LT<:Lattices.RealLattice
    state = spherer(LT, L; r, g1=0, g2=1)
    # all indices with distance m
    shell = Lattices.shell(state.lattice, r)
    g = 2
    for i in shell
        state[i] = g
        add_edge!(state.phylogeny, nv(state.phylogeny), 1)
        push!(state.meta[end, :snps], g)
        g += 1
    end
    state, shell
end

"""
    sphere_with_single_mutant_on_outer_shell(::Type{LT}, L::Int; r, s=1.0)

Fill a ball of radius `r` with genotype `1`. Place a single cell of genotype `2` on the outermost sphere at random.
"""
function sphere_with_single_mutant_on_outer_shell(::Type{LT}, L::Int; r, s=1.0) where LT<:Lattices.RealLattice
    state = spherer(LT, L; r, g1=0, g2=1)
    # all indices with distance r
    shell = Lattices.shell(state.lattice, r)
    
    g = 2
    i = rand(shell)
    state[i] = g
    add_edge!(state.phylogeny, nv(state.phylogeny), 1)
    push!(state.meta[end, :snps], g)

    state.meta[2, :fitness] = s
    state, shell, i
end


##--END module--
end
