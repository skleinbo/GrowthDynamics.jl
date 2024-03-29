module Observables

import Base.Iterators: filter
import CoordinateTransformations: Spherical, SphericalFromCartesian
import DataFrames: DataFrame, subset
import Dictionaries: dictionary, Dictionary, insert!
import Distributions: Multinomial
import GeometryBasics: Pointf, Point3f
import Graphs: SimpleGraph, SimpleDiGraph, nv, inneighbors
import Graphs: outneighbors, neighborhood, neighborhood_dists
import Graphs: vertices, enumerate_paths, bellman_ford_shortest_paths
import LinearAlgebra: dot, norm, Symmetric
import StatsBase: countmap, mean, sample, var, Weights
import ..Lattices
import ..Lattices: CubicLattice, RealLattice, Index, midpoint, midpointcoord, coord, index, isonshell
import ..Lattices: neighbors, neighbors!, Neighbors, out_of_bounds, spacings
import ..Populations: Population, index, hassnps

using ..Phylogenies
import ..Phylogenies: parent

export  allele_fractions,
        sampled_allele_fractions,
        allele_spectrum,
        total_population_size,
        population_size,
        surface,
        surface2,
        boundary,
        lone_survivor_condition,
        living_ancestor,
        # cphylo_hist,
        # phylo_hist,
        mrca,
        npolymorphisms,
        polymorphisms,
        common_snps,
        pairwise,
        mean_pairwise,
        popsize_on_shells,
        ipositions,
        positions,
        explode_into_shells,
        tajimasd,
        icompetition,
        counts_on_shells,
        counts_on_shells_vec

"Dictionary `(SNP, population count)`"
function allele_size(S::Population, t=0)
    X = Dict{Int, Int}()
    for v in S.meta
        isnothing(v.snps) && continue
        for snp in v.snps
            if haskey(X, snp)
                X[snp] += v.npop
            else
                push!(X, snp => v.npop)
            end
        end
    end
    return X
end

"""
    sampled_allele_fractions(S::Population[, t=0, samples=length(S.meta.npops)])

Randomly sample genotypes(!) and calculate frequencies of contained SNPs.
Return a dictionary `(SNP, freq)`.
"""
function sampled_allele_fractions(S::Population, samples=length(S.meta))
    X = Dict{eltype(eltype(S.meta.snps)), Float64}()
    T = total_population_size(S)
    pop_samples = sample(1:length(S.meta),
        Weights(S.meta[:, :npops] ./ T), samples)
    for j in pop_samples
        for snp in S.meta.snps[j]
            if haskey(X, snp)
                X[snp] += 1 / length(pop_samples)
            else
                push!(X, snp => 1 / length(pop_samples))
            end
        end
    end
    return X
end

"Dictionary of `(SNP, freq)`."
allele_fractions(S::Population, t=0) = begin
    as = allele_size(S, t)
    Dict(zip(keys(as), values(as) ./ total_population_size(S)))
end

function allele_fractions(L::Lattices.RealLattice{<:Integer})
    m = maximum(L.data)
    if m == 0
        return []
    end
    genotypes = 1:m
    Ntot = count(!=(0), L.data)
    Ng = [(g,0.) for g in genotypes]
    for g in genotypes
        Ng[g] = (g,count(x->x==g,L.data)/Ntot)
    end
    return Ng
end

"""
    allele_spectrum(state;[ threshold=0.0, read_depth=total_population_size(state)])

Return a DataFrame with count, frequency of every polymorphism. Additionally sample from the population.
"""
function allele_spectrum(state::Population; threshold=0.0, read_depth=total_population_size(state))
  popsize = total_population_size(state)
  ## Set state to analyse
  as = allele_size(state, 0)
  positions = collect(keys(as))
  npops = collect(values(as))
  as = DataFrame(position=positions, npop=npops)
  if isempty(as)
      return as
  end
  as.fpop = as.npop ./ popsize

  ## Detection threshold
  as = subset(as, :fpop => (x->x .>= threshold))

  ## Sampling
  sample_percent = read_depth / popsize
  # as.depth = rand(Binomial(popsize, sample_percent), size(as, 1))
  as.depth = fill(read_depth, size(as,1))

  if sample_percent < 1.0
    allele_sample_size = ceil(Int, sample_percent*size(as,1))
    as.samples = rand(Multinomial(round(Int,sample_percent*sum(as.npop)), as.npop/sum(as.npop)))
  else
    allele_sample_size = size(as,1)
    as.samples = as.npop
  end

  return as
end

function allele_spectrum(as::DataFrame; threshold=0.0, read_depth=sum(as.npop))
  # Recover population size
  popsize = round(Int, as[1, :npop] / as[1, :fpop])

  ## Detection threshold
  as = filter(x->x.fpop >= threshold, as) |> DataFrame

  ## Sampling
  sample_percent = read_depth / popsize
  # as.depth = rand(Binomial(popsize, sample_percent), size(as, 1))
  as.depth = fill(read_depth, size(as,1))

  if sample_percent < 1.0
    allele_sample_size = ceil(Int, sample_percent*size(as,1))
    as.samples = rand(Multinomial(round(Int,sample_percent*sum(as.npop)), as.npop/sum(as.npop)))
  else
    allele_sample_size = size(as,1)
    as.samples = as.npop
  end

  return as
end

function total_population_size(L::Lattices.RealLattice{<:Integer})
    count(!=(0), L.data)
end

"Total population size. Duh."
function total_population_size(S::Population)
    sum(@view S.meta[:, :npop])
end

function population_size(L::Lattices.RealLattice{T}, t) where T<:Integer
    D = Dict{T, Int}()
    for x in L.data
        if x==0
            continue
        end
        if !haskey(D, x)
            push!(D, x=>1)
        else
            D[x] += 1
        end
    end
    return sort( [ (k,v) for (k,v) in D ], lt=(x,y)->x[1]<y[1] )
end

"Dictionary (genotype, population size)"
function population_size(S::Population, t)
    zip(S.meta[:, :genotype], S.meta[:, :npop]) |> collect
end

function genotype_dict(S::Population{T, A}) where {T,A}
    return S.meta.index
end

function total_birthrate(S::Population{T, Lattices.RealLattice{T}}; baserate=1.0) where T
    L = S.lattice
    F = S.meta[:fitnesses]
    G = genotype_dict(S)
    nn = Lattices.empty_neighbors(L)
    total_rate = 0.0
    for j in CartesianIndices(L.data)
        if L.data[j] == 0
            continue
        end
        total_rate += (1.0 - Lattices.density!(nn, L, j))*baserate*F[G[L.data[j]]]
    end
    total_rate
end

function total_birthrate(S::Population{T, Lattices.NoLattice{T}}; baserate=1.0) where T
    sum(S.meta[:npops] .* S.meta[:fitnesses]) * baserate
end


#########################
## Spatial observables ##
#########################

function kpzroughness(S::AbstractDict, lattice; kwargs...)
    sort(Dict( t=>kpzroughness(S[t], lattice; kwargs...) for t in keys(S) ))
end

function kpzroughness(v::AbstractVector, lattice, args...;kwargs...)
    v = interface(v, lattice)[2]
    if length(v) < 2
        return 0.0
    end
    o = coord(lattice, midpoint(lattice))
    r = round(norm(v[1].-o))
    com = SphericalFromCartesian()(mean(v.-o))
    com = Spherical(r, com.θ, com.ϕ)
    geodists = map(v) do p
        r*acos( min(1.0, dot(p,o)/(norm(p)*norm(o))) )
    end
    # @show geodists
    var(geodists)
end 

function roughness(S::AbstractDict, args...; kwargs...)
    # (2*sqrt(pi)*sqrt(length(S[t])))
    sort(Dict( t=>linterface(S[t], args...; kwargs...)/sqrt(length(S[t])>0 ? length(S[t]) : 1 ) for t in keys(S) ))
end
function roughness(v::AbstractVector, args...;kwargs...)
    linterface(v, args...;kwargs...)
    # length(v)==1 ? zero(eltype(v[1])) : std( norm.(v .- com))
end
function linterface(S::AbstractDict, args...;kwargs...)
    sort(Dict( t=>linterface(S[t], args...; kwargs...) for t in keys(S) ))
end
function linterface(v::AbstractVector, args...; kwargs...)
    length(interface(v, args...; kwargs...)[1])
end

function interface(v::AbstractVector, state::Population; o=coord(state.lattice, midpoint(state.lattice)))
    lattice = state.lattice
    if length(v)==0
        return (CartesianIndex{3}[], Point3f[])
    end
    a = spacings(lattice)[1]
    ir = round(norm(v[1].-o)/a)
    r = round(norm(v[1].-o)/a)*a #round(norm(com-o))
    iv = Lattices.index.(Ref(lattice), v) # indices

    # "(not)-on-shell neighbors"
    nbs = Neighbors(lattice)
    inter = Base.filter(iv) do i
        neighbors!(nbs, lattice, i)
        nos_neighbors = filter(
            j->!out_of_bounds(lattice, j) && 
            (state[j]==0 || round(norm(coord(lattice, j)-o)/a)>ir),
        nbs) # not-on-shell-neighbors
        isempty(nos_neighbors) && return false

        any(nosn->any(
                    j->!out_of_bounds(lattice, j) &&
                        state[j]!=0 && state[j]!=state[i] &&
                        isonshell(lattice, coord(lattice, j), r, o),
                    neighbors(lattice, nosn)
                ),
            nos_neighbors
        )

    end
    cinter = coord(lattice, inter)
    return inter, cinter
end

interface(state::Population{G, <:RealLattice}, g::G) where G = interface(positions(state, g), state)

function interface(state::Population{G, <:RealLattice}, v::AbstractVector{<:Lattices.Index}) where G
    lat = state.lattice
    nn = Neighbors(lat)
    freenb = fill(false, size(nn))
    Base.filter(v) do I
        neighbors!(nn, lat, I)
        freenb = fill!(freenb, false)
        # filter out if no free neighboring site, i.e. if cannot grow
        # record empty neighboring sites
        bIgnore = true
        for (i,nb) in enumerate(nn)
            out_of_bounds(lat, nb) && continue
            if state[nb] == zero(G)
                bIgnore = false
                freenb[i] = true
            end
        end
        bIgnore && return false
        
        # Iterate over all empty neighboring sites
        # nn2: secondary neighbors == neighbors of empty neighbor
        for i in eachindex(nn)
            !freenb[i] && continue
            nn2 = neighbors(lat, nn[i])
            for n2 in nn2
                (n2 == I || n2 in v) && continue
                (I in neighbors(lat, n2)) && return true
            end
        end
        return false
    end
end

hasemptyneighbor(state::Population{G, A}, I::Index) where {G,A} = any(n->state[n]==zero(G), neighbors(state.lattice, I))

function icompetition(state::Population{G, <:RealLattice}, g::G) where G
    lat = state.lattice
    nv, ncomp = 0, 0 
    v = ipositions(state, g)

    # filter out those that cannot grow, i.e. have no empty neighboring sites
    filter!(v) do I
        hasemptyneighbor(state, I)
    end
    # collect all empty neighboring sites
    emptyneighbors = mapreduce(union, v) do I
        nn = neighbors(lat, I)
        emptynn = Base.filter(n->state[n]==zero(G), nn)
    end
    # create a mapping
    # empty site => (no. of neighbors of type `g`, no. of neighbors of any other type)
    # the counts of which are a measure for the competition between genotype g and 
    # all others for a given empty site.
    intf = map(emptyneighbors) do en
        nn = neighbors(lat, en)
        native = Base.filter(n->state[n]==g, nn)
        foreign = Base.filter(n->state[n]!=g && state[n]!=zero(G), nn)
        return en => (native, foreign)
    end
    # filter out entries with no competition
    filter!(intf) do (_, (_, foreign))
        !isempty(foreign)
    end
end

function surface(L::Lattices.RealLattice{<:Integer}, g::Int)
    x = 0
    I = CartesianIndices(L.data)
    for j in eachindex(L.data)
        if L.data[j]==g
            for n in Lattices.neighbors(L, I[j])
                if !Lattices.out_of_bounds(I[n], L.Na) && L.data[n]!=g && L.data[n]!=0
                    x+=1
                    break
                end
            end
        end
    end
    return x
end

function surface2(L::Lattices.RealLattice{<:Integer}, g::Int)
    x = 0
    y = 0
    I = CartesianIndices(L.data)
    for j in eachindex(L.data)
        if L.data[j]==g
            is_surface = false
            for n in Lattices.neighbors(L, I[j])
                if !Lattices.out_of_bounds(n, L.Na) && L.data[n]!=g && L.data[n]!=0
                    is_surface = true
                    y += 1
                end
            end
            if is_surface
                x += 1
            end
        end
    end
    return (x,ifelse(x>0,y/x,0.))
end

function boundary(L::Lattices.AbstractLattice{T, 1}, g) where T
    s =  count( x->x==g, view(L.data, 1) )
    s += count( x->x==g, view(L.data, L.Na) )
    return s
end

function boundary(L::Lattices.AbstractLattice{T, 2}, g) where T
    s =  count( x->x==g, view(L.data, :,1) )
    s += count( x->x==g, view(L.data, :,L.Nb) )
    s += count( x->x==g, view(L.data, 1,:) )
    s += count( x->x==g, view(L.data, L.Na,:) )
    return s
end

function boundary(L::Lattices.AbstractLattice{T, 3}, g) where T
    s =  count( x->x==g, view(L.data, :,:,1) )
    s += count( x->x==g, view(L.data, :,:,L.Nc) )
    s += count( x->x==g, view(L.data, :,1,:) )
    s += count( x->x==g, view(L.data, :,L.Nb,:) )
    s += count( x->x==g, view(L.data, 1,:,:) )
    s += count( x->x==g, view(L.data, L.Na,:,:) )
    return s
end

function lone_survivor_condition(L,g::Integer)
    af = allele_fractions(L)
    for x in af
        if x==(g,1.0)
            return true
        end
    end
    return false
end

function lone_survivor_condition(L)
    u = unique(L.data)
    return 0 in u && length(u)==2 || length(u)==1
end

## Helper function to merge dictionaries with vectors as values
## appends new value to old value
function mymerge(A,B)
    isempty(A) && return B
    isempty(B) && return A
    K = keys(A)
    C = map(collect(K)) do k
        if haskey(B, k)
            return k => vcat(A[k], B[k])
        else
            if typeof(A[k]) <: AbstractVector
                return k => A[k]
            else
                return k => [A[k]]
            end
        end
    end |> Dict
    K = keys(B)
    len = findmax(map(length,values(C)))[1] - 1
    len = max(len,0)
    # @info C
    # @info len
    try
        for k in K
            if !haskey(C, k)
                if typeof(B[k]) <: AbstractVector
                    push!(C, k => [zeros(eltype(B[k]), len); B[k]])
                else
                    push!(C, k => [zeros(eltype(B[k]), len); B[k]])
                end

            end
        end
    catch err
        @show A,B,C
        throw(err)
    end

    C
end

######################
# Positions & Shells #
######################

"""
    ipositions(state, g)

Returns lattice indices of cells of genotype `g`.
"""
function ipositions(state::Population, g)
    findall(==(g), state.lattice.data)
end

"""
    positions(state, g)

Returns coordinates of cells of genotype `g`.
"""
function positions(state::Population{T, <:Lattices.AbstractLattice{T,N}}, g) where {T,N}
    idx = ipositions(state, g)
    # dim = Lattices.dimension(state.lattice)
    convert(Vector{Pointf{N}}, map(I->Pointf{N}(Lattices.coord(state.lattice, I)), idx))
end

"""
    explode_into_shells(v, o, a; r0=)

Take a vector of cartesian coordinates `v`, center them around the midpoint `o`, and return
a dictionary `radius=>coordinates` where `r0<= radius <= max(||v||)` in increments of `a`. 
"""
function explode_into_shells(v::Vector{T}, o, a; r0=0f0) where T<:Pointf
    maxr = maximum(x->norm(x-o), v)
    Dict(r => Base.filter(x->r-a/2<=norm(x-o)<r+a/2, v) for r in r0:a:maxr+a)
end

function explode_into_shells(state::Population{G,A}, g::G,
    o=midpointcoord(state.lattice), a=spacings(state.lattice)[1];
    r0=0.0
) where {G,A}
    v = positions(state, g)
    maxr = maximum(x->norm(x-o), v; init=0.0)
    imaxr = Int(maxr÷a)
    dictionary(i => Base.filter(x->isonshell(state.lattice, x, i*a, o; a), v) for i in round(Int, r0÷a):(imaxr+1))
end
"""
    counts_on_shells(state, g, [o, a])

Return a dictionary `i=>count` mapping the `ith` shell to the number of cells
of genotype `g` on it.

The "shell number" of a point is the integer closest to `|p-o|/a`.

# Keyword arguments
* `o=midpointcoord(state.lattice)`: center of shells
* `a=spacing(state.lattice)`: spacing of shells
"""
function counts_on_shells(state::Population{G,A}, g::G,
    o=midpointcoord(state.lattice), a=spacings(state.lattice)[1];
) where {G,A}
    D = Dictionary{Int, Int}(;sizehint=maximum(size(state.lattice)))
    Cart = CartesianIndices(state.lattice.data)
    pos_iter = Iterators.map(i->coord(state.lattice, Cart[i]), Iterators.filter(i->state[i]==g, eachindex(state.lattice.data)))
    for p in pos_iter
        shell = round(Int, norm(p-o)/a)
        if haskey(D, shell)
            D[shell] += 1
        else
            insert!(D, shell, 1)
        end
    end
    D
end

function counts_on_shells_vec(state::Population{G,A}, g::G,
    o=midpointcoord(state.lattice), a=spacings(state.lattice)[1];
    maxr
) where {G,A}
    D = zeros(Int, maxr)
    Cart = CartesianIndices(state.lattice.data)
    pos_iter = Iterators.map(i->coord(state.lattice, Cart[i]) ,Iterators.filter(i->state[i]==g, eachindex(state.lattice.data)))
    for p in pos_iter
        shell = round(Int, norm(p-o)/a)
        if length(D)>shell
            D[shell+1] += 1
        else
            push!(D, 1)
        end
    end
    D
end

"""
    popsize_on_shells(T, outer, [o=midpoint(T.lattice)])

Creates a dictionary `genotype => trajectory` where `trajectory` is a vector of population size.
`trajectory[r]` is the the population size on a L2-norm shell of radius `r` around `o`.

`r` ranges between `1..outer` in increments of `a` (defaults to the lattice spacing).

Set `deleteone/zero=false` to keep the wildtype/count empty sites.
"""
function popsize_on_shells(state::Population{T, <:RealLattice}, outer; a=spacings(state.lattice)[1], deleteone=true, deletezero=true) where T
    dist_mat = Lattices.euclidean_dist_matrix(state.lattice, Lattices.midpoint(state.lattice))
    shell_inds = map(1:outer) do r; findall(x-> r-a/2 < x <= r+a/2, dist_mat ); end

    # aux. dictionary to store first/latest appearance of genotype
    genotypes = state.meta[:, :genotype]
    firstlast = Dict{T, Tuple{Int,Int}}()  #(map(g->g=>(typemax(Int),0), genotypes))

    ret = mapreduce((a,b)->mymerge(a,b), 1:outer) do r #radius
        inds = shell_inds[r]
        shell = state[inds]
        c = countmap(reshape(shell, :))
        if deletezero
            delete!(c, 0)
        end
        if deleteone
            delete!(c, 1)
        end
        for g in keys(c)
            newfirst = haskey(firstlast, g) ? firstlast[g][1] : r
            firstlast[g] = (newfirst, r)
        end
        for g in keys(firstlast)
            if !haskey(c, g)
                c[g] = 0
            end
        end
        c
    end

    for g in keys(ret)
        # pad trajectories
        ret[g] = vcat(
            zeros(Int, firstlast[g][1]-1),
            ret[g],
            zeros(Int, outer-firstlast[g][2])
            )
    end
    return ret

end


#############################
## Pylogenetic observables ##
#############################

"Earliest living ancestor."
function living_ancestor(S::Population, g)
    ancestor = parent(S, g)
    while ancestor !== nothing && S.meta[ancestor.id, :npop] == 0
        ancestor = parent(S, ancestor.g)
    end
    return ancestor
end

## TODO: Replace IndexedTables with DataFrames
#= function phylo_hist(state::Population)
    nb = neighborhood_dists(state.phylogeny, 1, nv(state.phylogeny), dir=:in)
    nb_table = table(map(nb) do x
        (state.meta[x[1], :genotype],x[2])
    end, pkey=[1])
    nb_table = renamecol(nb_table, 1=>:g, 2=>:dist)
    # dists = getindex.(nb,2)
    ps_table = table(population_size(state, 0), pkey=[1])
    ps_table = renamecol(ps_table, 1=>:g, 2=>:npop)

    # verts = getindex.(nb,1)
    join(ps_table, nb_table)
end

function cphylo_hist(state::Population)
    P = state.phylogeny
    nb = neighborhood_dists(P, 1, nv(P), dir=:in)
    ps_table = table(population_size(state, 0), pkey=[1])
    nb_table = table(map(nb) do x
        (state.meta[x[1], :genotype], x[1], x[2])
    end, pkey=[1])
    nb_table = rename(nb_table, 1=>:g, 2=>:nv, 3=>:dist)
    ps_table = rename(ps_table, 1=>:g, 2=>:npop)

    joined = join(nb_table, ps_table, how=:outer)
    joined = transform(joined, :cnpop => :npop)

    unP = SimpleGraph(P)
    paths = enumerate_paths(bellman_ford_shortest_paths(unP, 1))

    npops = select(joined, :npop)|>copy
    cnpops = select(joined, :cnpop)|>copy

    popfirst!(paths)
    for path in paths
        cnpop_new = cnpops[path[end]]
        cnpops[path[1:end-1]] .+= cnpop_new
    end

    return transform(joined, :cnpop => cnpops)
end =#


is_leaf(P::SimpleDiGraph, v) = length(inneighbors(P, v)) == 0

"""
    common_snps(::Population; filterdead=true)

List polymorphisms that are common to all genotypes.

# Optional arguments:
- `filterdead=true`: exclude unpopulated genotypes?
"""
function common_snps(S::Population; filterdead=true)
    firstpopulated = findfirst(v->v.npop > ifelse(filterdead, 0, -1), S.meta)
    if isnothing(firstpopulated)
        return Int[]
    end
    snps = copy(S.meta[firstpopulated[1], :snps])
    @show snps
    for v in S.meta
        filterdead && v.npop == 0 && continue
        isnothing(v.snps) && return Int[]
        intersect!(snps, v.snps)
        isempty(snps) && return Int[]
    end
    return snps
end

"""
    polymorphisms(S::Population)

Vector of polymorphisms (segregating sites).
"""
function polymorphisms(S::Population; filterdead=true)
    firstpopulated = findfirst(v->v.npop > ifelse(filterdead, 0, -1), S.meta)
    if isnothing(firstpopulated)
        return Set{Int}()
    end
    snps = hassnps(S.meta, firstpopulated) ? Set{Int}(S.meta[firstpopulated, :snps]) : Set{Int}()
    for v in S.meta
        filterdead && v.npop == 0 && continue
        isnothing(v.snps) && continue
        union!(snps, v.snps)
    end
    snps
end
"""
Return index of the most recent common ancestor between `(i,j)` in a phylogeny.
"""
function mrca(S::Population, i::Integer, j::Integer)
    P = S.phylogeny
    @assert 1<=i<=nv(P) && 1<=j<=nv(P)

    mrca = 1 ## 'Worst' case: MRCA is the root

    while i > 1 && j > 1
        (i == j) && return i ## (i,j) have coalesced -> return MRCA
        i = outneighbors(P, i)[1]
        # @info i,j
        (i == j) && return i ## (i,j) have coalesced -> return MRCA
        j = outneighbors(P, j)[1]
        # @info i,j
        (i == j) && return i ## (i,j) have coalesced -> return MRCA
    end

    return mrca
end

"""
Return index of the most recent common ancestor in a phylogeny.
"""
function mrca(S::Population)
    P = S.phylogeny

    _mrca = nv(P)

    idx = findall(>(0), (@view S.meta[:, Val(:npops)])) ## Only check the currently alive

    for i in idx[2:end], j in idx[2:end]
        _mrca = min(_mrca, mrca(S, i,j))
        _mrca == 1 && return _mrca
    end
    return _mrca
end
"""
    npolymorphisms(S::Population)

Number of polymrphisms
"""
npolymorphisms(S::Population) = length(polymorphisms(S))

function nsymdiff(A,B)
    x = 0
    A = sort(A)
    B = sort(B)

    if isempty(A)
        return length(B)
    elseif isempty(B)
        return length(A)
    end

    j = 1
    k = 1
    while j<=length(A) && k<=length(B)
        if A[j] == B[k]
            j += 1
            while 2<=j<=length(A) && A[j] == A[j-1]
                j += 1
            end
            k += 1
            while 2<=j<=length(B) && B[j] == B[j-1]
                k += 1
            end
        elseif A[j] > B[k]
            k += 1
            while 2<=j<=length(B) && B[j] == B[j-1]
                k += 1
            end
            x += 1
        else
            j += 1
            while 2<=j<=length(A) && A[j] == A[j-1]
                j += 1
            end
            x += 1
        end
    end
    x + length(A)-(j-1) + length(B)-(k-1)
end

"""
    pairwise(S::Population, i, j)

Number of pairwise genomic differences between genotype indices `i,j`.
"""
function pairwise(S::Population, i, j)
    si = S.meta[g=i, :snps]
    sj = S.meta[g=j, :snps]
    nsymdiff(si,sj)
end

"""
    pairwise(S::Population)

Matrix of pairwise differences.  
`filterdead`: Do not include extinct genotypes.
"""
function pairwise(S::Population;
     genotypes=S.meta[:, :genotype], 
     filterdead=false
)
    if filterdead
        itr = [ g for g in genotypes if S.meta[g=g, :npop]>0 ]
    else
        itr = genotypes
    end
    X = Matrix{Int}(undef, length(itr), length(itr))
    for (i,g1) in enumerate(itr), (j, g2) in enumerate(itr)
        if j<i
            continue
        end
        X[i,j] =  pairwise(S, g1, g2)
    end
    Symmetric(X)
end

"""
Diversity (mean pairwise difference of mutations) of a population.
"""
function mean_pairwise(S::Population)
    af = allele_fractions(S, 0)
    if length(af) > 0
        return mapreduce(x->2.0*x*(1-x), +, (af |> values |>collect))
    else
        return 0.0
    end
end

"""
See https://en.wikipedia.org/wiki/Tajima%27s_D
"""
function tajimasd(n, S, k)
    a1 = sum((1/i for i in 1:n-1))
    a2 = sum((1/i^2 for i in 1:n-1))
    b1 = (n+1)/3/(n-1)
    b2 = 2(n^2+n+3)/9/n/(n-1)
    c1 = b1 - 1/a1
    c2 = b2 - (n+2)/a1/n + a2/a1^2
    e1 = c1/a1
    e2 = c2/(a1^2+a2)

    return (k - S/a1) / sqrt(e1*S + e2*S*(S-1))
end

tajimasd(S::Population) = tajimasd(total_population_size(S), npolymorphisms(S), mean_pairwise(S))


##end module
end
