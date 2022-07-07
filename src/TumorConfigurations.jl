module TumorConfigurations

import Base: axes, checkbounds, copyto!, copy, eachindex, firstindex, getindex, IndexStyle, IndexLinear, lastindex
import Base: length, @propagate_inbounds, push!, resize!, setindex!, similar, size, show, view, zero
import Base.Iterators: product
import CoordinateTransformations: SphericalFromCartesian
import Dictionaries: Dictionary
import Graphs: SimpleDiGraph, add_vertex!, add_vertices!, add_edge!, nv
import GeometryBasics: Point2f0, Point3f0
import ..Lattices
import ..Lattices: AbstractLattice
import ..Lattices: coord, dimension, index, isonshell, radius, realsize, midpoint, dist, spacings
import ..Lattices: sitesperunitcell
import LinearAlgebra: norm
import StatsBase

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
default_metadatum() = (0, 1.0, Int64[], (0, 0.0))
"""
    MetaDatum

NamedTuple to store information about one genotype.
"""
function MetaDatum(A::MetaDatumFieldTypes)
    NamedTuple{MetaDatumFields}(A)
end

function MetaDatum(g)
    MetaDatum((g, default_metadatum()...))
end

mutable struct MetaData{T} <: AbstractArray{MetaDatum{T}, 1}
    _len::Int64
    index::Dictionary{T, Int}
    genotypes::Vector{T}
    npops::Vector{Int}
    fitnesses::Vector{Float64}
    snps::Vector{Vector{Int}}
    ages::Vector{Tuple{Int,Float64}}  ## (simulation t, real t) when a genotype entered.
    misc::Dict{Any,Any} # store anything else in here.
end

IndexStyle(::Type{<:MetaData}) = IndexLinear()

"""
    MetaData(T::DataType)

Empty MetaData for genotype-type T.
"""
MetaData(T::DataType) = MetaData(0, Dictionary{T, Int}(), T[], Int64[], Float64[], Vector{Int64}[], Tuple{Int64,Float64}[], Dict())

function MetaData{T}(::UndefInitializer, n) where T
    return MetaData{T}(0,
        Dictionary{T, Int}(),
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
MetaData(a::MetaDatumFieldTypes{T}) where {T} = MetaData{T}(1, Dictionary(a[1], 1), [a[1]], [a[2]], [a[3]], [a[4]], [a[5]], Dict())

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
    M = MetaData(N, Dictionary{T, Int}(), g, n, fitnesses, snps, ages, Dict())
    index!(M)
    M
end

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

_pluralize(field::Symbol) = _pluralize(Val(field))
_pluralize(::Val{:genotype}) = :genotypes
_pluralize(::Val{:npop}) = :npops
_pluralize(::Val{:snps}) = :snps
_pluralize(::Val{:fitness}) = :fitnesses
_pluralize(::Val{:age}) = :ages
_pluralize(::Val{F}) where F = throw(ArgumentError("Unknown field $F"))

@propagate_inbounds function gindex(M::MetaData{T}, g::T) where T
    return haskey(M.index, g) ? M.index[g] : nothing
end

@propagate_inbounds function Base.getindex(M::MetaData{T}; g) where {T}
    M[gindex(M, g)]
end

@propagate_inbounds function Base.getindex(M::MetaData{T}, i::Integer) where {T}
    @boundscheck if i>length(M)
        throw(BoundsError(M, i))
    end
    (genotype = M.genotypes[i], npop = M.npops[i], fitness = M.fitnesses[i], snps = M.snps[i], age = M.ages[i])
end
Base.getindex(M::MetaData{T}, ::Colon) where {T} = @inbounds M[eachindex(M)]

@propagate_inbounds function Base.getindex(M::MetaData{T}, I) where {T}
    @boundscheck checkbounds(Bool, M, I) || throw(BoundsError(M, I))
    N = MetaData{T}(
        0,
        filter(x->x in I, M.index),
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

Base.@propagate_inbounds getindex(M::MetaData, i::Integer, field::Symbol) = getindex(M, i, Val(field))

@propagate_inbounds function getindex(M::MetaData, i::Integer, field::Val{F}) where F
    mfield = _pluralize(field)
    getindex(getproperty(M, mfield), i)
end

@propagate_inbounds function getindex(M::MetaData{T}, field::Symbol; g::T) where T
    getindex(M, Val(field); g)
end

@propagate_inbounds function getindex(M::MetaData{T}, F::Val; g::T) where {T}
    getindex(M, gindex(M, g), F)
end

@propagate_inbounds function getindex(M::MetaData{T}, ::Colon, field::Symbol) where T
    return getproperty(M, field)[eachindex(M)]
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
    if length(M.genotypes) == n
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
    int_length = length(M.genotypes) # internal length
    if i >= int_length
        _resize!(M, ceil(Int64, max(1, int_length*2)))
    end
    M._len = i
    setindex!(M, D, i)
end

@propagate_inbounds Base.setindex!(M::MetaData, D::MetaDatum, i::Integer) = setindex!(M, values(D), i)

@propagate_inbounds function Base.setindex!(M::MetaData{T}, D::MetaDatumFieldTypes{T}, i::Integer) where {T}
    @boundscheck checkbounds(Bool, M, i) || throw(BoundsError(M, I))

    g = M.genotypes[i] = D[1]
    if isnothing(gindex(M, g))
        insert!(M.index, g, i)
    else
        M.index[g] = i
    end
    M.npops[i] = D[2]
    M.fitnesses[i] = D[3]
    M.snps[i] = D[4]
    M.ages[i] = D[5]
    D
end

@inline @propagate_inbounds function setindex!(M::MetaData, v, i::Integer, field)
    @boundscheck checkbounds(Bool, M, i) || throw(BoundsError(M, I))

    mfield = _pluralize(field)
    @inbounds setindex!(getproperty(M, mfield), v, i)
    return v
end

@inline @propagate_inbounds function setindex!(M::MetaData, v, field; g)
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

mutable struct TumorConfiguration{G, T <: Lattices.AbstractLattice}
    lattice::T
    phylogeny::SimpleDiGraph{Int64}
    meta::MetaData{G}
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

@propagate_inbounds function Base.setindex!(T::TumorConfiguration{S, <:AbstractLattice{S, A}}, v, ind::Vararg{Int64}) where {S,A}
    z = zero(S)
    L = T.lattice.data
    g_old = L[ind...]
    if g_old == v
        return v
    end
    if v != z
        @boundscheck begin
            if @inbounds isnothing(gindex(T.meta, v))
                throw(ArgumentError("Genotype $v is not know. Use push!(::TumorConfiguration, $v) first."))
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

"Add a new _unconnected_ genotype to a TumorConfiguration."
@propagate_inbounds @inline function Base.push!(S::TumorConfiguration{T, <:Lattices.TypedLattice{T}}, g::T) where {T}
    push!(S, MetaDatum{T}((g, 0, 1.0, Int64[], (S.t, S.treal))))
end

@propagate_inbounds function Base.push!(S::TumorConfiguration{T, <:Lattices.TypedLattice{T}}, M::MetaDatum{T}) where {T}
    @boundscheck if !isnothing(gindex(S.meta, M.genotype))
        throw(ArgumentError("genotype $(M.genotype) already present"))
    end
    push!(S.meta, M)
    add_vertex!(S.phylogeny)
    lastindex(S.meta)
end

### similar et al. ###
# // TODO: Generalize
Base.similar(C::TumorConfiguration) = TumorConfiguration(typeof(C.lattice)(C.lattice.a, zero(C.lattice.data)))

## Define getter/setter for all fields of MetaData
## Better than dynamically dispatching on getindex
## in performance critical code.
for field in MetaDatumFields
    getfn = Symbol("get",field)
    setfn = Symbol("set",field,"!")
    fieldplural = string(_pluralize(field))
    @eval begin
        export $getfn, $setfn
        function $getfn(M::MetaData{T}, g::T) where T
            id = gindex(M, g)
            @boundscheck if isnothing(id)
                throw(ArgumentError("Unknown genotype $g"))
            end
            getindex(getproperty(M, Symbol($fieldplural)), id)
        end
        $getfn(S::TumorConfiguration, g) = $getfn(S.meta, g)

        function $setfn(M::MetaData{T}, v, g::T) where T
            id = gindex(M, g)
            @boundscheck if isnothing(id)
                throw(ArgumentError("Unknown genotype $g"))
            end
            setindex!(getproperty(M, Symbol($fieldplural)), v, id)
        end
        $setfn(S::TumorConfiguration, v, g) = $setfn(S.meta, v, g)
    end
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
Populated with one individual of genotype `1::Int64` with fitness 1.0.
"""
function nolattice_state()
    state = TumorConfiguration(Lattices.NoLattice())
    push!(state, 1)
    state.meta[end, :npop] = 1
    state.meta[end, :fitness] = 1.0
    return state, nothing
end


"""
    uniform(::Type{LT<:RealLattice}, L; g=0, a=1.0)

Return system on a lattice of type `LT` with linear extension `L`, filled with genotype `g`.

# Example

    uniform(HexagonalLattice, 128; g=1)
"""
function uniform(T::Type{LT}, L::Int; g=0) where LT<:Lattices.RealLattice
    lattice = LT(1.0, fill(g, sitesperunitcell(LT, L)))
    state = TumorConfiguration(lattice)
    return state, nothing
end

"""
    single_center(::Type{RealLattice}, L; g1=1, g2=2)

Put a single cell of genotype `g2` at the midpoint of a lattice filled with `g1`.
"""
function single_center(::Type{LT}, L::Int; g1=0, g2=1) where LT<:Lattices.RealLattice
    state, _ = uniform(LT, L; g=g1)
    mid = midpoint(state.lattice)
    push!(state, g2)
    state[mid] = g2

    return state, mid
end

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
    spherer(::Type{LT}, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice

Fill lattice of type `LT` (e.g `CubicLattice`) with genotype `g1` and put a (L2-)sphere
of approx. radius `r` with genotype `g2` at the center.
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
    spheref(::Type{LT}, L::Int; r = 0, g1=0, g2=1) where LT<:Lattices.RealLattice

Fill lattice of type `LT` (e.g `CubicLattice`) with genotype `g1` and put a (L2-)sphere with genotype `g2`
that occupies approx. a fraction `f` of the lattice at the center.
"""
function spheref(::Type{LT}, L::Int; f = 1 / 10, g1=0, g2=1) where LT<:Lattices.RealLattice
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
    if g1!=0 && g2!=0
        add_edge!(state.phylogeny, g2, g1)
    end

    return state, sphere
end

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
    sphere_with_single_mutant_on_outer_shell(::Type{LT}, L::Int; r, s=1.0)

Fill a ball of radius `r` with genotype `1`. Place a single cell of genotype `2` on the outermost sphere at random.
"""
function sphere_with_single_mutant_on_outer_shell(::Type{LT}, L::Int; r, s=1.0) where LT<:Lattices.RealLattice
    state, _ = spherer(LT, L; r, g1=0, g2=1)
    # all indices with distance r
    shell = Lattices.shell(state.lattice, r)
    
    g = 2
    i = rand(shell)
    push!(state, g)
    state[i] = g
    add_edge!(state.phylogeny, nv(state.phylogeny), 1)
    push!(state.meta[end, :snps], g)

    state.meta[2, :fitness] = s
    state, shell, i
end


##--END module--
end
