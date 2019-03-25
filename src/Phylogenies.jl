module Phylogenies

using   Distributions,
        MetaGraphs,
        LightGraphs

import ..TumorConfigurations

export  annotate_snps!,
        df_traversal

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
Annotate a phylogeny with SNPs. Every vertex in the phylogeny inherits the SNPs its parent
plus (on average) `μ` new ones.

* `μ`: genome wide rate (Poisson) / count (uniform)
* `L=10^9`: length of the genome
* `allow_multiple=false`: Allow for a site to mutate more than once.
* `kind=:poisson`: `:poisson` or `:fixed`
"""
function annotate_snps!(S::TumorConfigurations.TumorConfiguration, μ;
    L=10^9, allow_multiple=false, kind=:poisson)

    P = S.Phylogeny
    D = Poisson(μ)

    tree = df_traversal(P.graph)
    set_prop!(P, 1, :snps, Int[])
    for v in tree
        parent = outneighbors(P, v)[1]
        snps = copy(get_prop(P, parent, :snps))
        if kind == :poisson
            count = rand(D)
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
        set_prop!(P, v, :snps, snps)
    end
end



end
