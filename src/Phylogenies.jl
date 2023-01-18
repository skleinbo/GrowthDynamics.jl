module Phylogenies

import Base: parent
using   Distributions,
        Graphs
import Base: parent
import ..Lattices: AbstractLattice

export  children,
        df_traversal,
        has_children,
        harm,
        harm2,
        isleaf,
        isroot,
        nchildren,
        parent,
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

isroot(::SimpleDiGraph, ::Nothing) = throw(ArgumentError("Not a valid vertex."))
isroot(P::SimpleDiGraph, v) = isempty(outneighbors(P, v))

parent(P::SimpleDiGraph, v) = outneighbors(P, v)[1]

children(P::SimpleDiGraph, v) = inneighbors(P, v)

nchildren(P::SimpleDiGraph, v) = length(children(P, v))

"Does a genotype have any children?"
has_children(S, g) = nchildren(S, g) > 0
isleaf(args...) = !has_children(args...)

end ## MODULE ##
