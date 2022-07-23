module Phylogenies

using   Distributions,
        Graphs
import Base: parent
import ..Lattices: AbstractLattice
import ..TumorConfigurations
import ..TumorConfigurations: getnpop, index, hassnps, TumorConfiguration

export  annotate_snps!,
        annotate_lineage!,
        add_snps!,
        children,
        df_traversal,
        has_children,
        harm,
        harm2,
        mrca,
        nchildren,
        prune_phylogeny!,
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

"""
    parent(S::TumorConfiguration, g)

Parent of a genotype `g`.  Return tuple `(id=index, g=genotype)`.
"""
function parent(S::TumorConfiguration, g)
    vertex = index(S.meta, g)
    n = outneighbors(S.phylogeny, vertex)
    if length(n)!=1
        return nothing
    end
    return (id=n[1], g=S.meta[n[1], :genotype])
end

"""
    children(S::TumorConfiguration, g)

Vector of direct descendants of a genotype.

!!! info
    Returns indices.
"""
function children(S::TumorConfiguration, g)
    vertex = index(S.meta, g)
    inneighbors(S.phylogeny, vertex)
end

"Number of direct descendends of a genotype."
nchildren(S::TumorConfiguration, g) = length(children(S, g))

"Does a genotype have any children?"
has_children(S, g) = nchildren(S, g) > 0

"""
    annotate\\_snps!(S::TumorConfiguration, μ;
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
function annotate_snps!(S::TumorConfiguration, μ;
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
    annotate\\_lineage!(S::TumorConfiguration, μ, v;
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
function annotate_lineage!(S::TumorConfiguration{T, <:AbstractLattice{T}}, μ, v::Int, root=1;
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

"""
    prune_phylogeny!(S::TumorConfiguration)

Remove unpopulated genotypes from the phylogenetic tree and meta data.  
Any gap in the phylogeny is bridged.
"""
function prune_phylogeny!(S::TumorConfigurations.TumorConfiguration{G,L}) where {G,L}
    P = S.phylogeny::SimpleDiGraph{Int64}

    function bridge!(s, d)
        children = inneighbors(P, d)
        if s==1 || getnpop(S, d) > 0
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
        while parent[1]!=1 && !isempty(parent) && S.meta[parent[1], Val(:npop)] == 0
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
    S.phylogeny = induced_subgraph(P, subvertices)[1]::SimpleDiGraph{Int64}
    S.meta = @inbounds S.meta[subvertices]
    return S.phylogeny, S.meta
end

"""
Return index of the most recent common ancestor between `(i,j)` in a phylogeny.
"""
function mrca(S::TumorConfiguration, i::Integer, j::Integer)
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
function mrca(S::TumorConfiguration)
    P = S.phylogeny

    _mrca = nv(P)

    idx = findall(>(0), (@view S.meta[:, Val(:npops)])) ## Only check the currently alive

    for i in idx[2:end], j in idx[2:end]
        _mrca = min(_mrca, mrca(S, i,j))
        _mrca == 1 && return _mrca
    end
    return _mrca
end



end ## MODULE ##
