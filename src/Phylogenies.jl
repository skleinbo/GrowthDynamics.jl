module Phylogenies

using   Distributions,
        Graphs
import Base: parent
import ..Lattices: AbstractLattice

export  annotate_snps!,
        annotate_lineage!,
        add_snps!,
        children,
        df_traversal,
        has_children,
        harm,
        harm2,
        isleaf,
        isroot,
        nchildren,
        sample_ztp

harm(N::Integer) = sum(1/i for i in 1:N)
harm2(N::Integer) = sum(1/i^2 for i in 1:N)

"""
    df_traversal(G::SimpleDiGraph; root=1)

Traverse a phylogenetic tree depth first, starting at vertex `root` which defaults to `1`.

Return a vector of vertices.
"""
function df_traversal(G::SimpleDiGraph; root=1)
    V = Int[]
    df_traversal!(V, G; root)
    V
end

"""
See also [`df_traversal`](@ref)
"""
function df_traversal!(V::Vector{Int}, G::SimpleDiGraph; root::Int=1)
    for v in inneighbors(G, root)
        push!(V, v)
        df_traversal!(V, G; root=v)
    end
end

"""
    sample_ztp(lambda)

Return one sample of a zero-truncated Poisson distribution with rate `λ`.

See [](https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution)

__Note:__ Should be in `Distributions.jl`.
"""
function sample_ztp(lambda::Float64)
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

isroot(P::SimpleDiGraph, ::Nothing) = throw(ArgumentError("Not a valid vertex."))
isroot(P::SimpleDiGraph, v) = isempty(outneighbors(P, v))

parent(P::SimpleDiGraph, v) = outneighbors(P, v)[1]

children(P::SimpleDiGraph, v) = inneighbors(P, v)

nchildren(P::SimpleDiGraph, v) = length(children(P, v))

"Does a genotype have any children?"
has_children(S, g) = nchildren(S, g) > 0
isleaf(args...) = !has_children(args...)



"""
Take a vector of SNPS and add new ones, or replace them. Typically called from
dynamics during a mutation event.

* `μ`: genome wide rate (Poisson) / count (uniform)
* `L=10^9`: length of the genome
* `allow_multiple=false`: Allow for a site to mutate more than once.
* `kind=:poisson` Either `:poisson` or `:fixed`
- `replace=false` Replace existing SNPs.
"""
function add_snps!(S::Vector, μ;
    L=10^9, allow_multiple=false, kind=:poisson, replace=false)

    if replace
        empty!(S)
    end

    if kind == :poisson
        count = sample_ztp(μ)
    else
        count = μ
    end

    if allow_multiple
        append!(S, rand(1:L, count))
    else # randomize `count` _new_ SNPs
        j = 0
        while j < count
            s = rand(1:L)
            if !(s in S)
                push!(S, s)
                j += 1
            end
        end
    end
    sort!(S)
    S
end




end ## MODULE ##
