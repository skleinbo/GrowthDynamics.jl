module Phylogenies

using   Distributions,
        MetaGraphs,
        LightGraphs

import ..TumorConfigurations

export  annotate_snps!,
        df_traversal,
        harm,
        harm2,
        prune_phylogeny

harm(N::Integer) = sum(1/i for i in 1:N)
harm2(N::Integer) = sum(1/i^2 for i in 1:N)

"""
Traverse a phylogenetic tree depth first, starting at vertex 1, which is
assumed to be the root.

Returns a vector of vertices.
"""
function df_traversal(G::SimpleDiGraph)
    V = Int[]
    df_traversal!(V, G)
    V
end

df_traversal!(V::Vector{Int}, G::SimpleDiGraph) = df_traversal!(V, G, 1)

function df_traversal!(V::Vector{Int}, G::SimpleDiGraph, r::Int)
    for v in inneighbors(G, r)
        push!(V, v)
        df_traversal!(V, G, v)
    end
end

"""
Zero-truncated Poisson sampler with rate `λ`.
"""
@inline function sample_ztp(lambda::Float64)
  k = 1
  t = exp(-lambda) / (1 - exp(-lambda)) * lambda
  s = t
  u = rand()
  while s < u
    k += 1
    t *= lambda / k
    s += t
  end
  k
end


"""
Annotate a phylogeny with SNPs. Every vertex in the phylogeny inherits the SNPs its parent
plus (on average) `μ` new ones.

* `μ`: genome wide rate (Poisson) / count (uniform)
* `L=10^9`: length of the genome
* `allow_multiple=false`: Allow for a site to mutate more than once.
* `kind=:poisson`: `:poisson` or `:fixed`
- `replace=false`: Replace existing SNPs.
"""
function annotate_snps!(S::TumorConfigurations.TumorConfiguration, μ;
    L=10^9, allow_multiple=false, kind=:poisson, replace=false)

    P = S.Phylogeny
    SNPS = S.meta.snps
    D = Poisson(μ)

    tree = df_traversal(P)
    # set_prop!(P, 1, :snps, Int[])
    for v in tree
        if !replace && !isempty(SNPS[v])
            continue
        end
        parent = outneighbors(P, v)[1]
        snps = copy(SNPS[parent])
        if kind == :poisson
            count = sample_ztp(μ)
        else
            count = μ
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
        SNPS[v] = snps
    end
end


"""
Remove unpopulated genotypes from the graph.
"""
function prune_phylogeny(S::TumorConfigurations.TumorConfiguration)
    P = copy(S.Phylogeny)
    npops = S.meta.npops

    function bridge!(s, d)
        children = inneighbors(P, d)
        if s==1 || npops[d] > 0
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

    itr = filter(v->npops[v]==0 && v!=1, df_traversal(P))|>collect
    subvertices = setdiff(1:nv(P), itr)
    for (i,v) in enumerate(itr)
        children = inneighbors(P, v)
        parent = outneighbors(P, v)
        @debug "Vertex $v is empty" v children  parent[1]
        while parent[1]!=1 && !isempty(parent) && npops[parent[1]] == 0
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
    SP = induced_subgraph(P, subvertices)
    return SP, S.meta[subvertices]
end
function prune_phylogeny(S::MetaDiGraph)
    P = copy(S)

    function bridge!(s, d)
        children = inneighbors(P, d)
        if s==1 || get_prop(P,d,:npop) > 0
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

    itr = filter(v->get_prop(P,v,:npop)==0 && v!=1, df_traversal(P.graph))|>collect
    subvertices = setdiff(1:nv(P), itr)
    for (i,v) in enumerate(itr)
        children = inneighbors(P, v)
        parent = outneighbors(P, v)
        @debug "Vertex $v is empty" v children  parent[1]
        while parent[1]!=1 && !isempty(parent) && get_prop(P, parent[1], :npop) == 0
            parent = outneighbors(P, parent[1])
            # if isempty(parent) || parent[1]==1
            #     break
            # end
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
    # while begin v=findfirst(x->get_prop())
    SP = induced_subgraph(P, subvertices)
    # @assert is_connected(SP[1])
    return SP
    # if !is_connected(SP[1])
    #     return (P,SP)
    # else
    #     SP
    # end
end




end
