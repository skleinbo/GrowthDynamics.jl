module TumorObservables

import IndexedTables: table, join, rename, transform, select, filter
import DataFrames: DataFrame
import LinearAlgebra: Symmetric
import StatsBase: Weights, sample
import Distributions: Multinomial


import LightGraphs: SimpleGraph,
                    SimpleDiGraph,
                    nv, inneighbors, neighborhood, neighborhood_dists,
                    vertices,
                    enumerate_paths,
                    bellman_ford_shortest_paths


import ..Lattices
import ..LatticeTumorDynamics

import ..TumorConfigurations: TumorConfiguration

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
        nchildren,
        has_children,
        cphylo_hist,
        phylo_hist,
        polymorphisms,
        npolymorphisms,
        common_snps,
        pairwise,
        mean_pairwise

"Dictionary `(SNP, population count)`"
function allele_size(S::TumorConfiguration, t=0)
    X = Dict{eltype(eltype(S.meta.snps)), Int64}()
    for j in 1:length(S.meta.snps)
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
function sampled_allele_fractions(S::TumorConfiguration, t=0, samples=length(S.meta.npops))
    X = Dict{eltype(eltype(S.meta.snps)), Int64}()
    T = total_population_size(S)
    pop_samples = sample(1:length(S.meta.genotypes),
        Weights(S.meta.npops ./ T), samples)
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
    sum(S.meta.npops)
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
    zip(S.meta.genotypes, S.meta.npops) |> collect
end

function genotype_dict(S::TumorConfiguration)
    G = Dict{eltype(S.meta.genotypes), Int64}()
    for (k,g) in enumerate(S.meta.genotypes)
        push!(G, g=>k)
    end
    return G
end

function total_birthrate(S::TumorConfiguration{<:Lattices.RealLattice}; baserate=1.0)
    L = S.lattice
    F = S.meta.fitnesses
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
    sum(S.meta.npops .* S.meta.fitnesses)
end


function surface(L::Lattices.RealLattice{<:Integer}, g::Int)
    x = 0
    I = CartesianIndices(L.data)
    for j in eachindex(L.data)
        if L.data[j]==g
            for n in Lattices.neighbors(L, I[j])
                if !LatticeTumorDynamics.out_of_bounds(I[n], L.Na) && L.data[n]!=g && L.data[n]!=0
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
                if !LatticeTumorDynamics.out_of_bounds(n, L.Na) && L.data[n]!=g && L.data[n]!=0
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

function boundary(L::Lattices.AbstractLattice1D{<:Any}, g)
    s =  count( x->x==g, view(L.data, 1) )
    s += count( x->x==g, view(L.data, L.Na) )
    return s
end
function boundary(L::Lattices.AbstractLattice2D{<:Any}, g)
    s =  count( x->x==g, view(L.data, :,1) )
    s += count( x->x==g, view(L.data, :,L.Nb) )
    s += count( x->x==g, view(L.data, 1,:) )
    s += count( x->x==g, view(L.data, L.Na,:) )
    return s
end
function boundary(L::Lattices.AbstractLattice3D{<:Any}, g)
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

## Pylogenic observables
##
##

"Number of direct descendends of a genotype."
function nchildren(S::TumorConfiguration, g)
    vertex = findfirst(x->x==g, S.meta.genotypes)
    length(inneighbors(S.phylogeny, vertex))
end

"Does a genotype have any children?"
has_children(S, g) = nchildren(S, g) > 0

function phylo_hist(state::TumorConfiguration)
    nb = neighborhood_dists(state.phylogeny, 1, nv(state.phylogeny), dir=:in)
    nb_table = table(map(nb) do x
        (get_prop(state.phylogeny, x[1], :genotype),x[2])
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
        (state.meta.genotypes[x[1]], x[1], x[2])
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
end


is_leaf(P::SimpleDiGraph, v) = length(inneighbors(P, v)) == 0

"List polymorphisms that are common to all genotypes."
function common_snps(S::TumorConfiguration)
    populated = findall(v->v > 0, S.meta.npops)
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
    SNPS = Set(S.meta.snps[1])
    for v in vertices(S.phylogeny)
        if !is_leaf(S.phylogeny, v)
            continue
        end
        union!(SNPS, S.meta.snps[v])
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
    si = S.meta.snps[i]
    sj = S.meta.snps[j]
    nsymdiff(si,sj)
end

"""
    pairwise(S::TumorConfiguration)

Matrix of pairwise differences.
"""
function pairwise(S::TumorConfiguration)
    X = fill(0, nv(S.phylogeny), nv(S.phylogeny))
    for i in 1:nv(S.phylogeny), j in i+1:nv(S.phylogeny)
        X[i,j] = pairwise(S, i, j)
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
