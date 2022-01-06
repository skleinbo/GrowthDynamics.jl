module TumorObservables

# import IndexedTables: table, join, rename, renamecol, transform, select, filter
import DataFrames: DataFrame
import LinearAlgebra: norm, Symmetric
import StatsBase: Weights, sample, countmap
import Distributions: Multinomial
import GeometryBasics: Pointf0


import Graphs: SimpleGraph,
                    SimpleDiGraph,
                    nv, inneighbors, outneighbors, neighborhood, neighborhood_dists,
                    vertices,
                    enumerate_paths,
                    bellman_ford_shortest_paths


import ..Lattices

import ..TumorConfigurations: TumorConfiguration, gindex

using ..Phylogenies

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
        polymorphisms,
        npolymorphisms,
        common_snps,
        pairwise,
        mean_pairwise,
        extract_shells,
        positions,
        explode_into_shells

"Dictionary `(SNP, population count)`"
function allele_size(S::TumorConfiguration, t=0)
    X = Dict{eltype(eltype(S.meta.snps)), Int64}()
    for j in 1:length(S.meta)
        for snp in S.meta.snps[j]
            if haskey(X, snp)
                X[snp] += S.meta.npops[j]
            else
                push!(X, snp => S.meta.npops[j])
            end
        end
    end
    return X
end

"Dictionary of `(SNP, freq)`."
allele_fractions(S::TumorConfiguration, t=0) = begin
    as = allele_size(S, t)
    Dict(zip(keys(as), values(as) ./ total_population_size(S)))
end

"""
    sampled_allele_fractions(S::TumorConfiguration[, t=0, samples=length(S.meta.npops)])

Randomly sample genotypes(!) and calculate frequencies of contained SNPs.
Return a dictionary `(SNP, freq)`.
"""
function sampled_allele_fractions(S::TumorConfiguration, samples=length(S.meta))
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


function allele_fractions(L::Lattices.RealLattice{<:Integer})
    m = maximum(L.data)
    if m == 0
        return []
    end
    genotypes = 1:m
    Ntot = countnz(L.data)
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
function allele_spectrum(state::TumorConfiguration; threshold=0.0, read_depth=total_population_size(state))
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
  as = filter(x->x.fpop >= threshold, as) |> DataFrame

  ## Sampling
  sample_percent = read_depth / popsize
  # as.depth = rand(Binomial(popsize, sample_percent), size(as, 1))
  as.depth = fill(read_depth, size(as,1))

  if sample_percent < 1.0
    allele_sample_size = ceil(Int64, sample_percent*size(as,1))
    as.samples = rand(Multinomial(round(Int64,sample_percent*sum(as.npop)), as.npop/sum(as.npop)))
  else
    allele_sample_size = size(as,1)
    as.samples = as.npop
  end

  return as
end

function allele_spectrum(as::DataFrame; threshold=0.0, read_depth=total_population_size(state))
  # Recover population size
  popsize = round(Int, as[1, :npop] / as[1, :fpop])

  ## Detection threshold
  as = filter(x->x.fpop >= threshold, as) |> DataFrame

  ## Sampling
  sample_percent = read_depth / popsize
  # as.depth = rand(Binomial(popsize, sample_percent), size(as, 1))
  as.depth = fill(read_depth, size(as,1))

  if sample_percent < 1.0
    allele_sample_size = ceil(Int64, sample_percent*size(as,1))
    as.samples = rand(Multinomial(round(Int64,sample_percent*sum(as.npop)), as.npop/sum(as.npop)))
  else
    allele_sample_size = size(as,1)
    as.samples = as.npop
  end

  return as
end



function total_population_size(L::Lattices.RealLattice{<:Integer})
    countnz(L.data)
end

"Total population size. Duh."
function total_population_size(S::TumorConfiguration)
    sum(@view S.meta.npops[begin:S.meta._len])
end

function population_size(L::Lattices.RealLattice{T}, t) where T<:Integer
    D = Dict{T, Int64}()
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
function population_size(S::TumorConfiguration, t)
    zip(S.meta[:, :genotypes], S.meta[:, :npops]) |> collect
end

function genotype_dict(S::TumorConfiguration)
    G = Dict{eltype(S.meta.genotypes), Int64}()
    for (k,g) in enumerate(S.meta[:genotypes])
        push!(G, g=>k)
    end
    return G
end

function total_birthrate(S::TumorConfiguration{<:Lattices.RealLattice}; baserate=1.0)
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

function total_birthrate(S::TumorConfiguration{<:Lattices.NoLattice}; baserate=1.0)
    sum(S.meta[:npops] .* S.meta[:fitnesses])
end


#########################
## Spatial observables ##
#########################


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
    try
        for k in K
            if !haskey(C, k)
                if typeof(B[k]) <: AbstractVector
                    push!(C, k => B[k])
                else
                    push!(C, k => [B[k]])
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
    function positions(state, g)

Returns coordinates of cells of genotype `g`.
"""
function positions(state::TumorConfiguration{<:Lattices.RealLattice}, g)
    idx = findall(x->x==g, state.lattice.data)
    dim = Lattices.dimension(state.lattice)
    convert(Vector{Pointf0{dim}}, map(I->Pointf0{dim}(Lattices.coord(state.lattice, I)), idx))
end

"""
    function explode_into_shells(v, mid, a; a0=)

Take a vector of cartesian coordinates `v`, center them around the midpoint `mid`, and return
a dictionary `radius=>coordinates` where `a0<= radius <= max(||v||)` in increments of `a`. 
"""
function explode_into_shells(v::Vector{T}, o, a; a0=0f0) where T<:Pointf0
    maxr = maximum(x->norm(x-o), v)
    Dict(r => filter(x->r-a/2<norm(x-o)<=r+a/2, v) for r in a0:a:maxr+a)
end

"""
    extract_shells(A::TumorConfiguration{<:Lattices.CubicLattice}, outer)

Explode tumor configuration into shells [r, r+a) with radii between r in [1..outer].
Return dictionary with trajectory of every genotype but 0 and 1 (set `deleteone=false` to keep the wildtype).
"""
function extract_shells(state::TumorConfiguration{<:Lattices.CubicLattice}, outer; deleteone=true)
    a = Lattices.spacings(state.lattice)[1]
    dist_mat = Lattices.euclidean_dist_matrix(state.lattice, Lattices.midpoint(state.lattice))
    shell_inds = map(1:outer) do r; findall(x-> r-a < x <= r, dist_mat ); end

    mapreduce((a,b)->mymerge(a,b), 1:outer) do r #radius
        inds = shell_inds[r]
        shell = state[inds]
        c = countmap(reshape(shell, :))
        # @show c
        delete!(c, 0)
        if deleteone
            delete!(c, 1)
        end
        c
    end
end


#############################
## Pylogenetic observables ##
#############################

"First living ancestor."
function living_ancestor(S::TumorConfiguration, g)
    ancestor = parent(S, g)
    while ancestor !== nothing && S.meta[ancestor.id, :npop] == 0
        ancestor = parent(S, ancestor.g)
    end
    return ancestor
end

## TODO: Replace IndexedTables with DataFrames
#= function phylo_hist(state::TumorConfiguration)
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

function cphylo_hist(state::TumorConfiguration)
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

"List polymorphisms that are common to all genotypes."
function common_snps(S::TumorConfiguration)
    populated = findall(v->v > 0, S.meta[:npops])
    if isempty(populated)
        return Int64[]
    else
        intersect(map(populated) do v
            try
                S.meta.snps[v]
            catch
                @info "Vertex $v carries no field snps."
                Int64[]
            end
        end...)
    end
end

"""
    polymorphisms(S::TumorConfiguration)

Vector of polymorphisms (segregating sites).
"""
function polymorphisms(S::TumorConfiguration)
    SNPS = Set(S.meta[1, :snps])
    for v in vertices(S.phylogeny)
        if !is_leaf(S.phylogeny, v)
            continue
        end
        union!(SNPS, S.meta[v, :snps])
    end
    SNPS
end

"""
    npolymorphisms(S::TumorConfiguration)

Number of polymrphisms
"""
npolymorphisms(S::TumorConfiguration) = length(polymorphisms(S))

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
    pairwise(S::TumorConfiguration, i, j)

Number of pairwise genomic differences between genotype indices `i,j`.
"""
function pairwise(S::TumorConfiguration, i, j)
    si = S.meta[g=i, :snps]
    sj = S.meta[g=j, :snps]
    nsymdiff(si,sj)
end

"""
    pairwise(S::TumorConfiguration)

Matrix of pairwise differences.  
`skipdead`: Do not include extinct genotypes.
"""
function pairwise(S::TumorConfiguration;
     genotypes=S.meta[:genotypes], 
     skipdead=false
)
    if skipdead
        itr = [ g for g in genotypes if S.meta[g=g, :npop]>0 ]
    else
        itr = genotypes
    end
    X = Matrix{Int64}(undef, length(itr), length(itr))
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
function mean_pairwise(S::TumorConfiguration)
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

tajimasd(S::TumorConfiguration) = tajimasd(total_population_size(S), npolymorphisms(S), mean_pairwise(S))


##end module
end
