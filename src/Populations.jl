module Populations

import Base: axes, checkbounds, copyto!, copy, dotview
import Base: eachindex, firstindex, getindex, IndexStyle, IndexLinear, lastindex, materialize!
import Base: length, @propagate_inbounds, push!, resize!, setindex!, similar, size, show, view, zero
import Base.Broadcast: Broadcasted, BroadcastStyle
import Base.Iterators: product
import CoordinateTransformations: SphericalFromCartesian
import DataFrames: rename!
import Dictionaries: Dictionary, index
import Graphs: SimpleDiGraph, add_vertex!, add_vertices!, add_edge!, induced_subgraph
import Graphs: add_edge!, inneighbors, nv, outneighbors, rem_vertex!
import GeometryBasics: Point2f, Point3f
import ..Lattices
import ..Lattices: AbstractLattice, RealLattice, TypedLattice
import ..Lattices: coord, dimension, index, isonshell, radius, realsize, midpoint, dist, spacings
import ..Lattices: sitesperunitcell
import LinearAlgebra: norm
import ..Phylogenies: children, df_traversal, isroot, isleaf, nchildren, parent, sample_ztp
import StatsBase

export add_genotype!, add_snps!, connect!, half_space, hassnps, lastgenotype, nolattice_state, MetaData, MetaDatum, index!
export push!, remove_genotype!, single_center, spheref, spherer, sphere_with_diverse_outer_shell
export sphere_with_single_mutant_on_outer_shell, Population, prune_phylogeny!, rename!, uniform

const SNPSType = Union{Nothing, Vector{Int}}

##-- METADATA for efficiently storing population information --##
const MetaDatumFields = (:genotype, :npop, :fitness, :snps, :age)
const MetaDatumFieldTypes{T,S<:SNPSType} = Tuple{T,Int,Float64,S,Tuple{Int,Float64}}

"""
    MetaDatum{T,S}

`NamedTuple` to store information about a single genotype of type `T` that
carries mutations of type `S` (default `Int`).
"""
const MetaDatum{T,S} = NamedTuple{MetaDatumFields,MetaDatumFieldTypes{T,S}}
default_metadatum() = (0, 1.0, nothing, (0, 0.0))

function MetaDatum(A::MetaDatumFieldTypes)
    NamedTuple{MetaDatumFields}(A)
end

function MetaDatum(g)
    MetaDatum((g, default_metadatum()...))
end

"""
    MetaData{T}

Indexable structure to store information about a population.

# Access

Use `meta[id, :field]` or `meta[g=G, :field]` to access and set information for genotype `G`,
or the genotype with id `id` respectively.

`id` is also the number of the vertex in the phylogeny corresponding to a given genotype.

`:field` is one of
* `:npop` - population size
* `:fitness` - fitness value
* `:genotype` - genotype corresponding to a given id
* `:snps`: - vector of integers representing mutations a genotype carries. 
  `nothing` if no mutations are present.
* `:age`: tuple of `(timestep, realtime)` of when the genotype was first instantiated.
Useful for putting lengths on the branches of a phylogeny.

Additionally, a global field `meta.misc::Dict{Any,Any}` exists to store arbitrary, user-defined
information.

See also [`MetaDatum`](@ref)
"""
mutable struct MetaData{T} <: AbstractArray{MetaDatum{T,S} where S<:SNPSType, 1}
    _len::Int
    index::Dictionary{T, Int}
    genotype::Vector{T}
    npop::Vector{Int}
    fitness::Vector{Float64}
    snps::Vector{SNPSType}
    age::Vector{Tuple{Int,Float64}}  ## (simulation t, real t) when a genotype entered.
    misc::Dict{Any,Any} # store anything else in here.
end

IndexStyle(::Type{<:MetaData}) = IndexLinear()

"""
    MetaData(T::DataType)

Empty `MetaData`` for genotype-type `T`.
"""
MetaData(T::DataType) = MetaData(0, Dictionary{T, Int}(), T[], Int[], Float64[],
 SNPSType[], Tuple{Int,Float64}[], Dict())

function MetaData{T}(::UndefInitializer, n) where T
    return MetaData{T}(0,
        Dictionary{T, Int}(),
        Vector{T}(undef, n),
        Vector{Int}(undef, n),
        Vector{Float64}(undef, n),
        Vector{SNPSType}(undef, n),
        Vector{Tuple{Int,Float64}}(undef, n),
        Dict()
    )
end

"""
    MetaData(M::Union{MetaDatum, Tuple})

Construct `MetaData` from a single datum. Argument can be an appropriate tuple or named tuple.

See [`MetaDatum`](@ref)
"""
MetaData(M::MetaDatum) = MetaData(values(M))
MetaData(a::MetaDatumFieldTypes{T}) where {T} = MetaData{T}(1, Dictionary(a[1], 1), [a[1]], [a[2]], [a[3]], [a[4]], [a[5]], Dict())

"""
    MetaData(g::Vector{T}, n::Vector{<:Integer})

Construct MetaData from vectors of genotypes and population sizes.
* Fitnesses default to 1.0
* SNPs default to `nothing`.
* Ages default to (0, 0.0)
* misc defaults to an empty dictionary.
"""
function MetaData(g::Vector{T}, n::Vector{<:Integer}) where {T}
    N = length(n)
    if length(g) != N
        throw(ArgumentError("Lengths of arguments do not match."))
    end
    fitnesses = fill(1.0, N)
    snps = SNPSType[nothing for _ in 1:N]
    ages = fill((0, 0.0), N)
    M = MetaData(N, Dictionary{T, Int}(), g, n, fitnesses, snps, ages, Dict())
    index!(M)
    M
end

snpsfrom(M::MetaData, g) = snpsfrom(M[g=g, Val(:snps)])
snpsfrom(::Nothing) = Int[]
snpsfrom(v::Vector{Int}) = copy(v)

hassnps(M::MetaData, v) = !isnothing(M[v, Val(:snps)]) && !isempty(M[v, Val(:snps)])
"""
    hassnps(M::MetaData; g)

Return `true` if genotype contains mutations.
"""
hassnps(M::MetaData; g) = !isnothing(M[g; Val(:snps)]) && !isempty(M[g; Val(:snps)])

"""
    lastgenotype(M::MetaData)

Return genotype that was added last.
"""    
lastgenotype(M::MetaData) = M[end, Val(:genotype)]

"""
    index!(::MetaData)

Reindex the metadata.
"""
function index!(M::MetaData{T}) where T
    empty!(M.index)
    for i in eachindex(M)
        insert!(M.index, M[i, Val(:genotype)], i)
    end
    return M.index
end

length(M::MetaData) = M._len

firstindex(::MetaData) = 1
lastindex(M::MetaData) = length(M)
eachindex(M::MetaData) = firstindex(M):lastindex(M)

size(M::MetaData) = (length(M),)

function Base.similar(M::MetaData{T}) where T
    return similar(M, M._len)
end
function Base.similar(::MetaData{T}, len::Int) where T
    return MetaData{T}(undef, len)
end

@propagate_inbounds function Base.copyto!(dest::MetaData{T}, src::S) where {T, S<:MetaData{T}}
    @boundscheck if length(src) > length(dest.genotype)
        throw(BoundsError())
    end
    dest._len = src._len
    dest.misc = copy(src.misc)
    for field in setdiff(fieldnames(S), [:_len, :misc, :index])
        copyto!(getproperty(dest, field), getproperty(src, field))
    end
    dest.index = copy(src.index)
    return dest
end

"""
    index(M, g)

Return the index of genotype `g`, or `nothing` if `g` is not
in meta data `M`.
""" 
@propagate_inbounds function index(M::MetaData{T}, g) where T
    return haskey(M.index, g) ? M.index[g] : nothing
end

"""
    getindex(M::MetaData; g)

Synonymous with `M[g=g] == M[;g]`.

Return a named tuple with metdata about genotype `g`. 
"""
@propagate_inbounds function Base.getindex(M::MetaData{T}; g) where {T}
    M[index(M, g)]
end

@propagate_inbounds function Base.getindex(M::MetaData{T}, i::Integer) where {T}
    @boundscheck if i>length(M)
        throw(BoundsError(M, i))
    end
    (genotype = M.genotype[i], npop = M.npop[i], fitness = M.fitness[i], snps = M.snps[i], age = M.age[i])
end
Base.getindex(M::MetaData{T}, ::Colon) where {T} = @inbounds M[eachindex(M)]

@propagate_inbounds function Base.getindex(M::MetaData{T}, I) where {T}
    @boundscheck checkbounds(Bool, M, I) || throw(BoundsError(M, I))
    N = MetaData{T}(
        0,
        filter(x->x in I, M.index),
        M.genotype[I],
        M.npop[I],
        M.fitness[I],
        M.snps[I],
        M.age[I],
        M.misc
    )
    N._len = length(N.genotype)
    index!(N)
    return N
end

Base.@propagate_inbounds getindex(M::MetaData, i::Integer, field::Symbol) = getindex(M, i, Val(field))

@propagate_inbounds function getindex(M::MetaData, i::Integer, ::Val{F}) where F
    # mfield = _pluralize(field)
    getindex(getproperty(M, F), i)
end

@propagate_inbounds function getindex(M::MetaData{T}, field::Symbol; g::T) where T
    getindex(M, Val(field); g)
end

@propagate_inbounds function getindex(M::MetaData{T}, F::Val; g::T) where {T}
    getindex(M, index(M, g), F)
end

@propagate_inbounds function getindex(M::MetaData{T}, ::Colon, field::Symbol) where T
    return getproperty(M, field)[eachindex(M)]
end

@propagate_inbounds function getindex(M::MetaData{T}, ::Colon, ::Val{F}) where {T,F}
    return getproperty(M, F)[eachindex(M)]
end

@propagate_inbounds function view(M::MetaData{T}, ::Colon, field::Val{F}) where {T, F}
    return Base.view(getproperty(M, F), eachindex(M))
end
Base.@propagate_inbounds view(M::MetaData{T}, ::Colon, field::Symbol) where {T} = view(M, :, Val(field))

@propagate_inbounds function getindex(M::MetaData{T}, I::AbstractVector, field::Symbol) where T
    @boundscheck checkbounds(Bool, M, I) || throw(BoundsError(M, I))
    return getfield(M, field)[I]
end

@propagate_inbounds function view(M::MetaData{T}, I::AbstractVector, field::Symbol) where T
    @boundscheck checkbounds(Bool, M, I) || throw(BoundsError(M, I))
    return Base.view(getfield(M, field), I)
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
    if length(M.genotype) == n
        return M
    end
    for field in fieldnames(MetaData)
        if field in [:_len, :misc, :index]
            continue
        else
            resize!(getproperty(M, field), n)
        end
    end

    M
end    

Base.push!(M::MetaData{T}, g::T) where T = push!(M, MetaDatum(g))
Base.push!(M::MetaData, D::MetaDatum) = push!(M, values(D))
@inline @propagate_inbounds function Base.push!(M::MetaData{T}, D::MetaDatumFieldTypes{T}) where T
    # @boundscheck if D[1] == zero(T) || D[1] in @view M.genotypes[begin:M._len]
    #     throw(ArgumentError("Invalid genotype $(D[1]): either already present or zero."))
    # end
    i = lastindex(M)+1
    int_length = length(M.genotype) # internal length
    if i >= int_length
        _resize!(M, ceil(Int, max(1, int_length*2)))
    end
    M._len = i
    setindex!(M, D, i)
end

@propagate_inbounds Base.setindex!(M::MetaData, D::MetaDatum, i::Integer) = setindex!(M, values(D), i)

@propagate_inbounds function Base.setindex!(M::MetaData{T}, D::MetaDatumFieldTypes{T}, i::Integer) where {T}
    @boundscheck checkbounds(Bool, M, i) || throw(BoundsError(M, i))

    g = M.genotype[i] = D[1]
    if isnothing(index(M, g))
        insert!(M.index, g, i)
    else
        M.index[g] = i
    end
    M.npop[i] = D[2]
    M.fitness[i] = D[3]
    M.snps[i] = D[4]
    M.age[i] = D[5]
    D
end

@inline @propagate_inbounds function setindex!(M::MetaData, v, i::Integer, ::Val{field}) where field
    @boundscheck checkbounds(Bool, M, i) || throw(BoundsError(M, i))
    # update index if genotype changes
    if field == :genotype
        old_g = M[i, Val(:genotype)]
        insert!(M.index, v, i)
        delete!(M.index, old_g)
    end
    @inbounds setindex!(getproperty(M, field), v, i)
    return v
end
@inline @propagate_inbounds function setindex!(M::MetaData, v, i::Integer, field)
    setindex!(M, v, i, Val(field))
end

@inline @propagate_inbounds function setindex!(M::MetaData, v, field; g)
    i = index(M, g)
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


##--  BEGIN Population                                   --##

## Certain values on the lattice are special. 
## For example, we need a way to identify the empty site.
## We use `zero` for that. If zero is undefined for the 
## type you are using, define it, e.g
zero(::Type{String}) = "0"

"""
    Population{G, T}

Represents a population on a lattice of type `T` with genotypes of data type `G`.

# Fields

* `lattice<:Lattices.AbstractLattice`
* `phylogeny`: directed graph recording the ancestry of genotypes
* `meta::MetaData`: metadata such as fitnesses, mutations, etc. See [`MetaData`](@ref).
* `t::Int`: age in timesteps
* `treal::Float64`: age in "real" time.
* `observables`: dictionary to store observables in
"""
mutable struct Population{G, T <: Lattices.AbstractLattice}
    lattice::T
    phylogeny::SimpleDiGraph{Int}
    meta::MetaData{G}
    t::Int
    treal::Float64
    observables::Dict{Symbol, Any}
end

"""
    Population(lattice, [phylogeny])

Wraps an existing lattice in a Population. Calculates meta.npops automatically.

If `phylogeny` is not given, it defaults to an empty graph.
"""
function Population(lattice::Lattices.RealLattice{T}, phylogeny::SimpleDiGraph = SimpleDiGraph()) where {T}
    counts_dict = StatsBase.countmap(lattice.data, alg = :dict)
    if haskey(counts_dict, zero(T))
        delete!(counts_dict, zero(T))
    end
    genotypes = collect(keys(counts_dict))
    npops = collect(values(counts_dict))
    metadata = MetaData(genotypes, npops)
    add_vertices!(phylogeny, length(genotypes))
    Population(lattice, phylogeny, metadata, 0, 0.0, Dict{Symbol,Any}())
end
function Population(nolattice::Lattices.NoLattice{T}) where T
    metadata = MetaData(T)
    phylogeny = SimpleDiGraph()
    Population(nolattice, phylogeny, metadata, 0, 0.0, Dict{Symbol,Any}())
end

connect!(T::Population, p::Pair{Int,Int}) = connect!(T, p[1], p[2])
connect!(T::Population, a::Int, b::Int) = add_edge!(T.phylogeny, a, b)

@propagate_inbounds Base.getindex(T::Population, ind...) = getindex(T.lattice.data, ind...)

@propagate_inbounds Base.setindex!(T::Population, v, ind::CartesianIndex) = setindex!(T, v, Tuple(ind)...)

@propagate_inbounds function Base.setindex!(T::Population{S, <:AbstractLattice{S, A}}, v, ind::Vararg{Int}) where {S,A}
    z = zero(S)
    L = T.lattice.data
    g_old = L[ind...]
    if g_old == v
        return v
    end
    if v != z
        @boundscheck begin
            if @inbounds isnothing(index(T.meta, v))
                throw(ArgumentError("Genotype $v is not know. Use push!(::Population, $v) first."))
            end
        end
        @inbounds T.meta[g=v, Val(:npop)] += 1
    end
    if g_old != z
        @inbounds T.meta[g=g_old, Val(:npop)] -= 1
    end
    L[ind...] = v
    v
end

## Broadcasting

function dotview(S::Population, I...)
    (S, I)
end

@inline function materialize!(::BroadcastStyle, dest::Tuple{S,U},
    bc::Broadcasted{Style}) where {Style, S<:Population, U}
    if bc.f !== identity
        throw(ArgumentError("Broadcasting functions other than `=` over Population is not implemented"))
    end
    state, idx = dest
    g = bc.args[1]
    idx = LinearIndices(state.lattice.data)[idx...]
    for I in idx
        state[I] = g      
    end
    state
end

"""
    add_genotype!(S::Population, G, parent)

Add genotype `G` to the population and connect it to `parent` in the phylogeny.
`G` is either a genotype or a full `MetaDatum`.
`parent` defaults to the first genotype, i.e. the root of the phylogenetic tree.
If `parent=nothing`, the genotype will not be connected to the tree.

See also: [`MetaDatum`](@ref), [`remove_genotype!`](@ref)
"""
add_genotype!(S::Population, G, parent=S.meta[1, :genotype]; kwargs...) = push!(S, G, parent; kwargs...)

"Add a new _unconnected_ genotype to a Population."
@propagate_inbounds @inline function Base.push!(S::Population{T, <:TypedLattice{T}}, g::T, args...) where {T}
    push!(S, MetaDatum{T, Nothing}((g, 0, 1.0, nothing, (S.t, S.treal))), args...)
end

@propagate_inbounds function Base.push!(S::Population{T, <:TypedLattice{T}}, M::MetaDatum{T}, parent=nothing) where {T}
    @boundscheck if !isnothing(index(S.meta, M.genotype))
        throw(ArgumentError("genotype $(M.genotype) already present"))
    end
    push!(S.meta, M)
    add_vertex!(S.phylogeny)
    inew = lastindex(S.meta)
    if !isnothing(parent)
        iparent = index(S.meta, parent)
        add_edge!(S.phylogeny, inew, iparent)
    end
    inew
end

"""
    remove_genotype_from_phylogeny!(P, v; bridge=true)

Remove vertex `v` from phylogeny `P`. If `bridge=true` (default), the gap
is closed by a new edge.
"""
function remove_genotype_from_phylogeny!(P::SimpleDiGraph, v; bridge=true, force=false)
    if isroot(P, v) 
        bridge = false 
        if !force
            throw(ArgumentError("Trying to remove the root. Override with `force=true` if you are certain."))
        end
    elseif isleaf(P, v)
        bridge = false
    end
    if bridge
        C = children(P, v)
        p = parent(P, v)
        foreach(c->add_edge!(P, c, p), C)
    end
    rem_vertex!(P, v)

    return true
end

function remove_genotype_from_metadata!(M::MetaData{T}, g::T) where {T}
    v = index(M, g)
    if isnothing(v)
        return false
    end
    ## Mimic behavior for removal from Graphs:
    ## Overwrite the meta data at index v with 
    ## those from the last position and trim.
    g_new = M[end, :genotype]
    M[v] = M[end]
    M._len -= 1
    M.index[g_new] = v
    delete!(M.index, g)
    return true
end

"""
    remove_genotype!(S::Population, g; bridge=true)

Remove genotype from the population. Discards it from meta data, prunes it from the 
phylogeny, and sets all corresponding sites of the lattice to zero.

If `bridge=true` (default), the gap in the phylogeny is bridged with new edges.

Throw an exception if the requested genotype does not exist.

Return `true` if successful, else `false`.
"""
function remove_genotype!(S::Population{T, <:Lattices.TypedLattice{T}}, g::T; bridge=true) where {T}
    v = index(S, g)
    isnothing(v) && throw(ArgumentError("Genotype $g does not exist."))
    if S.lattice isa RealLattice
        sites = S.lattice.data .== g 
        S.lattice.data[sites] .= zero(T)
    end
    return remove_genotype_from_phylogeny!(S.phylogeny, v) && remove_genotype_from_metadata!(S.meta, g)
end

"""
    rename!(S, g1 => g2)

Rename genotype `g1` to `g2`.
"""
function rename!(S::Population{T, <:Lattices.TypedLattice{T}}, x::Pair{T,T}) where T
    rename!(S.meta, x)
    if S.lattice isa RealLattice
        S.lattice.data[ S.lattice.data.==x[1] ] .= x[2]
    end
    nothing
end
function rename!(M::MetaData{T}, x::Pair{T,T}) where T
    @boundscheck if x[2] in M.index
        throw(ArgumentError("cannot rename to existing genotype $(x[2])"))
    end
    M[g=x[1], Val(:genotype)] = x[2]
    nothing
end

### similar et al. ###
# // TODO: Generalize
Base.similar(C::Population) = Population(typeof(C.lattice)(C.lattice.a, zero(C.lattice.data)))

## Define getter/setter for all fields of MetaData
## Better than dynamically dispatching on getindex
## in performance critical code.
for field in MetaDatumFields
    getfn = Symbol("get",field)
    setfn = Symbol("set",field,"!")
    _field = Meta.quot(field)
    @eval begin
        export $getfn, $setfn
        function $getfn(M::MetaData{T}, g::T) where T
            id = index(M, g)
            @boundscheck if isnothing(id)
                throw(ArgumentError("Unknown genotype $g"))
            end
            getindex(getproperty(M, $_field), id)
        end
        $getfn(S::Population, g) = $getfn(S.meta, g)

        function $setfn(M::MetaData{T}, v, g::T) where T
            id = index(M, g)
            @boundscheck if isnothing(id)
                throw(ArgumentError("Unknown genotype $g"))
            end
            setindex!(getproperty(M, Symbol($field)), v, id)
        end
        $setfn(S::Population, v, g) = $setfn(S.meta, v, g)
    end
end

add_edge!(state::Population, newgenotype, parent) = add_edge!(state.phylogeny, index(state, newgenotype), index(state, parent))

index(S::Population, args...) = index(S.meta, args...)

"Genotype that was last added to the population."
lastgenotype(S::Population) = lastgenotype(S.meta)

"Number of direct descendends of a genotype."
nchildren(S::Population, g) = length(children(S, g))

size(S::Population, args...) = size(S.lattice, args...)

"""
    children(S::Population, g)

Vector of direct descendants of a genotype.

!!! info
    Returns indices.
"""
function children(S::Population, g)
    vertex = index(S.meta, g)
    inneighbors(S.phylogeny, vertex)
end

isroot(S::Population, g) = isroot(S.phylogeny, index(S, g))
"""
    parent(S::Population, g)

Parent of a genotype `g`.  Return tuple `(id=index, g=genotype)`.
"""
function parent(S::Population, g)
    vertex = index(S.meta, g)
    n = outneighbors(S.phylogeny, vertex)
    if length(n)!=1
        return nothing
    end
    return (id=n[1], g=S.meta[n[1], :genotype])
end

"""
    add_snps!(state::Population, g, μ)

Randomize and add mutations to genotype `g`, or replace them.
`μ` is the genome wide mutation rate (Poisson) or mutation count (fixed).

__Note:__ `allow_multiple` is much faster if the number of mutations
already present is large.

## Keyword arguments

* `L=10^9`: length of the genome
* `allow_multiple=false`: allow for a site to mutate more than once.
* `kind=:poisson`: either `:poisson` or `:fixed`
- `replace=false`: replace existing SNPs.
"""
function add_snps!(state::Population, g, args...; kwargs...)
    state.meta[g=g, :snps] = snpsfrom(state.meta, g)
    add_snps!(state.meta[g=g, :snps], args...; kwargs...)

    return state[g=g, :snps] 
end

"""
    add_snps!(state::Population, g, v::Vector{Int})

Add mutations `v` to genotype `g`.
No checks for duplications are performed. 

Return a vector of all mutations. 
"""
function add_snps!(state::Population, g, v::Vector{Int})
    state.meta[g=g, :snps] = snpsfrom(state.meta, g)
    append!(state.meta[g=g, :snps], v)

    return state.meta[g=g, :snps] 
end


function add_snps!(S::Vector{Int}, μ;
    L=10^9, allow_multiple=false, count=:poisson, replace=false)

    if replace
        empty!(S)
    end

    if count === :poisson
        n = sample_ztp(μ)
    else
        n = μ
    end

    if allow_multiple
        append!(S, rand(1:L, n))
    else # randomize `n` _new_ SNPs
        j = 0
        while j < n
            s = rand(1:L)
            if !(s in S)
                push!(S, s)
                j += 1
            end
        end
    end
    sort!(S)

    return S
end

"""
    annotate\\_snps!(S::Population, μ;
        [L, allow_multiple=false, kind=:poisson, replace=false])

Annotate a phylogeny with SNPs. Every vertex in the phylogeny inherits the SNPs
of its parent, plus (on average) `μ` new ones.  
Skips any vertex that is already annotated, unless `replace` is set to `true`.

* `μ`: genome wide rate (Poisson) / count (uniform)
* `L=10^9`: length of the genome
* `allow_multiple=false`: Allow for a site to mutate more than once.
* `kind=:poisson` Either `:poisson` or `:fixed`
* `replace=false` Replace existing SNPs.
"""
function annotate_snps!(S::Population, μ;
    L=10^9, allow_multiple=false, kind=:poisson, replace=false)

    P = S.phylogeny
    M = S.meta
    # D = Poisson(μ)

    tree = df_traversal(P)
    # set_prop!(P, 1, :snps, Int[])
    for v in tree
        if !replace && hassnps(S.meta, v)
            continue
        end
        parent = outneighbors(P, v)[1]
        snps = hassnps(S.meta, parent) ? copy(M[parent, :snps]) : Int[]
        if kind == :poisson
            count = sample_ztp(μ)
        else
            count = μ
        end
        if count == 0
            continue
        end
        if allow_multiple
            append!(snps, rand(1:L, count))
        else # randomize `count` _new_ SNPs
            j = 0
            while j < count
                s = rand(1:L)
                if !(s in snps)
                    push!(snps, s)
                    j += 1
                end
            end
        end
        sort!(snps)
        @debug "Setting SNPs for $v"
        M[v, :snps] = snps
    end
end

"""
    annotate\\_lineage!(S::Population, μ, v;
        [L, allow_multiple=false, kind=:poisson, replace=false])

Annotate a _lineage_ (path from `v` to `root`) with SNPs. Every vertex in the phylogeny inherits the SNPs
of its parent, plus (on average) `μ` new ones.  
Skips any vertex that is already annotated, unless `replace` is set to `true`.

Ends prematurely if a vertex with annotation is found on the way from tip to root.

* `v``: vertex
* `root`: begin of lineage. Defaults to root (1) of tree.
* `μ`: genome wide rate (Poisson) / count (uniform)
* `L=10^9`: length of the genome
* `allow_multiple=false`: Allow for a site to mutate more than once.
* `kind=:poisson`: `:poisson` or `:fixed`
* `replace=true`: Replace existing SNPs.

!!! note
    If `replace` is `false`, any existing annotation will break the inheritance 
    from root to target vertex.

"""
function annotate_lineage!(S::Population{T, <:AbstractLattice{T}}, μ, v::Int, root=1;
    L=10^9, allow_multiple=false, kind=:poisson, replace=true) where {T}
    path = []
    while !isnothing(v) && v!=root # && (isnothing(S.meta[v, :snps]) || isempty(S.meta[v, :snps]))
        push!(path, v)
        p = outneighbors(S.phylogeny, v)
        v = isempty(p) ? nothing : p[1]
    end
    reverse!(path)
    psnps = Int[]
    for v in path
        if !isnothing(S.meta[v, :snps]) && !isempty(S.meta[v, :snps])
            S.meta[v, :snps] = add_snps!(psnps, μ; L, allow_multiple, kind, replace=true)           
        end 
        psnps = copy(S.meta[v, :snps])
    end
    return path
end

"""
    prune_phylogeny!(S::Population)

Remove unpopulated genotypes from the phylogenetic tree and meta data.  
Any gap in the phylogeny is bridged.
"""
function prune_phylogeny!(S::Population{G,L}) where {G,L}
    P = S.phylogeny::SimpleDiGraph{Int}

    function bridge!(s, d)
        children = inneighbors(P, d)
        if s==1 || S.meta[d, :npop] > 0
            @debug "Adding edge"  d s
            add_edge!(P, d, s)
        elseif length(children)==0
            return
        elseif length(children) >= 1
            for child in children
                bridge!(s, child)
            end
        end
    end

    itr = filter(v->S.meta[v, Val(:npop)]==0 && v!=1, df_traversal(P))|>collect
    subvertices = setdiff(1:nv(P), itr)
    for (i,v) in enumerate(itr)
        children = inneighbors(P, v)
        parent = outneighbors(P, v)
        @debug "Vertex $v is empty" v children  parent[1]
        while !isempty(parent) && parent[1]!=1 && S.meta[parent[1], Val(:npop)] == 0
            parent = outneighbors(P, parent[1])
        end
        if isempty(parent)
            continue
        end
        if !isempty(children)
            for child in children
                bridge!(parent[1], child)
            end
        end
        # @debug "Removing vertex" v
        # rem_vertex!(P, v)
    end
    S.phylogeny = induced_subgraph(P, subvertices)[1]::SimpleDiGraph{Int}
    S.meta = @inbounds S.meta[subvertices]
    return S.phylogeny, S.meta
end

###################################################################
## -- Convenience constructors for various initial geometries -- ##
###################################################################
#= Some of these methods return additional useful meta-data like
the indices of a shell. To be consistent, any such convinience
constructor must return a tuple `(state, meta)` where `meta` may
be `nothing`. =#

"""
    nolattice_state()

Model without spatial structure.  
Populated with one individual of genotype `1` with fitness 1.0.
"""
function nolattice_state()
    state = Population(Lattices.NoLattice())
    push!(state, 1)
    state.meta[end, :npop] = 1
    state.meta[end, :fitness] = 1.0
    return state, nothing
end


"""
    uniform(T, L; g=0, a=1.0)
    
A population on a lattice of type `T` with linear extension `L`, filled with genotype `g`.

Return `(population, nothing)`.

# Example

```jldoctest
julia> uniform(HexagonalLattice, 128; g=1)
(HexagonalLattice{Int64, Matrix{Int64}}
1	genotypes
16641	population, nothing)
```
"""
function uniform(T::Type{LT}, L::Int; g=0) where LT<:Lattices.RealLattice
    lattice = LT(1.0, fill(g, sitesperunitcell(LT, L)))
    state = Population(lattice)
    return state, nothing
end

"""
    single_center(T, L; g1=1, g2=2)

A single cell of genotype `g2` at the midpoint of a lattice of type `T` filled with `g1`.

Return `(population, midpoint::Index)`
"""
function single_center(::Type{LT}, L::Int; g1=0, g2=1) where LT<:Lattices.RealLattice
    state, _ = uniform(LT, L; g=g1)
    mid = midpoint(state.lattice)
    push!(state, g2)
    state[mid] = g2

    return state, mid
end

"""
    half_space(T, L; g1=1, g2=2)

A population of genotype `g1` on a lattice of type `T`. 
Fill the last dimension with `g2` up to fraction `f`

Return `(population, fill_to)` with `fill_to` defined as
`state[:,..., 1:fill_to] == g2` and `state[:,..., fill_to+1:end] == g1`.

# Example

```jldoctest
julia> using GrowthDynamics.Populations

julia> state = half_space(CubicLattice, 32, f=1/4, g1=1, g2=2)[1]
CubicLattice{Int64, Array{Int64, 3}}
2	genotypes
35937	population

julia> state.meta
2-element MetaData{Int64}:
 (genotype = 1, npop = 27225, fitness = 1.0, snps = nothing, age = (0, 0.0))
 (genotype = 2, npop = 8712, fitness = 1.0, snps = nothing, age = (0, 0.0))
```
"""
function half_space(::Type{LT}, L::Int; f=1/2, g1=0, g2=1) where LT<:Lattices.RealLattice
    if !(0.0<=f<=1)
        throw(ArgumentError("filling fraction must be between 0 and 1."))
    end
    state, _  = uniform(LT, L; g=g1)
    fill_to = round(Int, f*size(state.lattice)[end])
    fill_to >= 1 && push!(state, g2)
    for ind in product(axes(state.lattice.data)[1:end-1]..., 1:fill_to)
        state[ind...] = g2
    end

    return state, fill_to
end

"""
    spherer(T, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice

Fill lattice of type `T` (e.g `CubicLattice`) with genotype `g1` and put a (L2-)ball
of approx. radius `r` with genotype `g2` at the center.

Return `(population, idx_ball)` with `idx_ball` being the indices of sites within the ball.
"""
function spherer(::Type{LT}, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice
    if r < 0
        throw(ArgumentError("radius must be positive."))
    end

    state, _ = uniform(LT, L; g=g1)
    if g2==g1 || r==0.0
        return state, nothing
    end

    lat = state.lattice
    a = spacings(lat)[1] / 2
    midx = midpoint(lat)
    mid = coord(lat, midx)

    ## To prevent checking the distance for every lattice site,
    ## preselect the range of indices such that 
    ## -r-1 <= (x-o)_i <= r+1

    idx_ranges = UnitRange.(Tuple(index(lat, mid.-r.-4a)) , Tuple(index(lat, mid.+r.+4a)))
    idx_sphere = CartesianIndex.(collect(Iterators.filter(product(idx_ranges...)) do I
        p = coord(lat, I)
        norm(p-mid)<=r+a
    end))
    !isempty(idx_sphere) && push!(state, g2)
    foreach(idx_sphere) do I
        state[I] = g2
    end
    if g1!=0 && g2!=0
        add_edge!(state.phylogeny, g2, g1)
    end

    return state, idx_sphere
end

"""
    spheref(T, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice

Fill lattice of type `T` (e.g `CubicLattice`) with genotype `g1` and put a (L2-)ball with genotype `g2`
that occupies approx. a fraction `f` of the lattice at the center.

Return `(population, idx_ball)` with `idx_ball` being the indices of sites within the ball.
"""
function spheref(::Type{LT}, L::Int; f = 1 / 10, g1::G=0, g2::G=1) where {G,LT<:Lattices.RealLattice}
    if !(0.0<=f<=1)
        throw(ArgumentError("f must be between 0 and 1."))
    end
    state, _ = uniform(LT, L; g=g1)
    if g2==g1 || f==0.0
        return state, nothing
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
    !isempty(sphere) && push!(state, g2)
    for ind in sphere
        state[ind] = g2
    end
    if g1!=zero(G) && g2!=zero(G)
        add_edge!(state.phylogeny, index(state.meta, g2), index(state.meta, g1))
    end

    return state, sphere
end

"""
    sphere_with_diverse_outer_shell(T, L; r)

A lattice of type `T` (e.g `CubicLattice`) with an (L2-)ball of genotype `1`
and radius `r` at the center. Each site on the outermost shell is populated consecutively with 
a different genotype, starting from `2`.

Return `(population, idx_shell)` with `idx_shell` being the indices of the outer ("diverse") shell.
"""
function sphere_with_diverse_outer_shell(::Type{LT}, L::Int; r) where LT<:Lattices.RealLattice
    state, _ = spherer(LT, L; r, g1=0, g2=1)
    # all indices with distance m
    shell = Lattices.shell(state.lattice, r)
    g = 2
    for i in shell
        push!(state, g)
        state[i] = g
        add_edge!(state.phylogeny, nv(state.phylogeny), 1)
        push!(state.meta[end, :snps], g)
        g += 1
    end
    state, shell
end

"""
    sphere_with_single_mutant_on_outer_shell(T, L::Int; r, s=1.0)

A lattice of type `T` (e.g `CubicLattice`) with an (L2-)ball of genotype `1`
and radius `r` at the center. One random site on the outermost shell is populated 
with genotype `2` that has fitness `s`.

Return `(population, (idx_shell, idx_mutant))` with `idx_shell` being
the indices of the outer shell, and `idx_mutant` the index
of the mutant genotype.
"""
function sphere_with_single_mutant_on_outer_shell(::Type{LT}, L::Int; r, s=1.0) where LT<:Lattices.RealLattice
    state, _ = spherer(LT, L; r, g1=0, g2=1)
    # all indices with distance r
    shell = Lattices.shell(state.lattice, r)
    
    g = 2
    i = rand(shell)
    add_genotype!(state, g)
    state[i] = g
    add_snps!(state, g, [g])

    state.meta[2, :fitness] = s
    state, (shell, i)
end


##--END module--
end
