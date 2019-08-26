module TumorObservables

import IndexedTables: table, join, rename, transform, select
import LinearAlgebra: Symmetric

import OpenCLPicker: @opencl

@opencl import OpenCL
import MetaGraphs:  nv, inneighbors, neighborhood, neighborhood_dists,
                    set_prop!, get_prop, vertices,
                    filter_vertices
import LightGraphs: SimpleGraph,
                    enumerate_paths,
                    bellman_ford_shortest_paths


import ..Lattices
import ..LatticeTumorDynamics

import ..TumorConfigurations: TumorConfiguration

using ..Phylogenies

@opencl import ..OffLattice
@opencl import ..OffLatticeTumorDynamics

export  allelic_fractions,
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

function allelic_fractions(S::TumorConfiguration, t)
    X = Dict{eltype(eltype(S.meta.snps)), Int64}()
    for j in 1:length(S.meta.genotypes)
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

function allelic_fractions(L::Lattices.RealLattice{<:Integer})
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
@opencl function allelic_fractions(L::OffLattice.FreeSpace{<:Integer})
    state = L.genotypes[Bool.(L._mask)]
    m = maximum(state)
    if m == 0
        return []
    end
    genotypes = 1:m
    Ntot = countnz(state)
    Ng = [(g,0.) for g in genotypes]
    for g in genotypes
        Ng[g] = (g,count(x->x==g,state)/Ntot)
    end
    return Ng
end

function total_population_size(L::Lattices.RealLattice{<:Integer})
    countnz(L.data)
end

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

function population_size(S::TumorConfiguration, t)
    zip(S.meta.genotypes, S.meta.npops) |> collect
end

function set_population_size!(state::TumorConfiguration)
    if typeof(state.lattice) == NoLattice
        return
    end
    P = state.Phylogeny
    ps = population_size(state.lattice, 0)
    for (g,s) in ps
        set_prop!(P, P[g, :genotype], :npop, s)
    end
    nothing
end

## SLOW
@opencl function population_size(L::OffLattice.FreeSpace{<:Integer}, t)
    state = L.genotypes[Bool.(L._mask)]
    m = 0
    try
        m = maximum(state)
    catch
        return [(1,0.)]
    end
    if m == 0
        return [(1,0.)]
    end
    genotypes = 1:m
    Ng = [(g,0.) for g in genotypes]
    for g in genotypes
        Ng[g] = (g,count(x->x==g,state))
    end
    return Ng
end

function surface(L::Lattices.RealLattice{<:Integer}, g::Int)
    x = 0
    I = CartesianIndices(L.data)
    for j in eachindex(L.data)
        if L.data[j]==g
            for n in Lattices.neighbours(L, I[j])
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
            for n in Lattices.neighbours(L, I[j])
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

## REQUIRE OpenCL for FreeSpace
@opencl begin
function surface2(FS::OffLattice.FreeSpace{<:Integer}, t, g::Integer, sigma)
    # buf = OpenCL.cl.Buffer(Int32, OffLattice.cl_ctx, (:w,:alloc), FS.MaxPopulation)
    OpenCL.cl.set_args!(OffLattice.surface_kernel,
            FS._dMat_buf,
            FS._genotypes_buf,
            FS._mask_buf,
            FS._int_buf,
            FS.MaxPopulation,
            Int32(g),
            Float32(sigma)
    )
    OpenCL.cl.enqueue_kernel(OffLattice.cl_queue, OffLattice.surface_kernel, FS.MaxPopulation,min(FS.MaxPopulation,OffLattice.max_work_group_size))|>
        OpenCL.cl.wait
    Svec = OpenCL.cl.read(OffLattice.cl_queue, FS._int_buf)
    S0 = mapreduce(sign,+,0,Svec)|>Float64
    (S0, ifelse(S0>0.,sum(Svec)/S0,0.))
end

function surface2CPU(FS::OffLattice.FreeSpace{<:Integer}, g::Integer, sigma)
    s = 0
    @inbounds for j in 1:FS.MaxPopulation
        if FS._mask[j]==0 || FS.genotypes[j]!=g
            continue
        end
        for k in 1:FS.MaxPopulation
            if FS._mask[k]==0 || FS.genotypes[k]==g || norm(FS.positions[:,j]-FS.positions[:,k])>sigma
                continue
            end
            s +=1
        end
    end
    s
end

end
## END requires OpenCL

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
    af = allelic_fractions(L)
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
function nchildren(L,g)
    vertex = L.Phylogeny[:genotype][g]
    length(inneighbors(L.Phylogeny, vertex))
end

has_children(L,g) = nchildren(L,g) > 0

function phylo_hist(state::TumorConfiguration)
    nb = neighborhood_dists(state.Phylogeny, 1, nv(state.Phylogeny), dir=:in)
    nb_table = table(map(nb) do x
        (get_prop(state.Phylogeny, x[1], :genotype),x[2])
    end, pkey=[1])
    nb_table = renamecol(nb_table, 1=>:g, 2=>:dist)
    # dists = getindex.(nb,2)
    ps_table = table(population_size(state, 0), pkey=[1])
    ps_table = renamecol(ps_table, 1=>:g, 2=>:npop)

    # verts = getindex.(nb,1)
    join(ps_table, nb_table)
end

function cphylo_hist(state::TumorConfiguration)
    P = state.Phylogeny
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

## Phylogenetic observables

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

function polymorphisms(S::TumorConfiguration)
    union(S.meta.snps...) |> unique |> sort
end

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


function pairwise(S::TumorConfiguration, i, j)
    si = S.meta.snps[i]
    sj = S.meta.snps[j]
    nsymdiff(si,sj)
end

function pairwise(S::TumorConfiguration)
    X = fill(0, nv(S.Phylogeny), nv(S.Phylogeny))
    for i in 1:nv(S.Phylogeny), j in i+1:nv(S.Phylogeny)
        X[i,j] = pairwise(S, i, j)
    end
    Symmetric(X)
end

function mean_pairwise(S::TumorConfiguration)
    X = 0
    for i in 2:nv(S.Phylogeny), j in i:nv(S.Phylogeny)
        X += pairwise(S, i, j)*S.meta.npops[i]*S.meta.npops[j]
    end
    X / binomial(total_population_size(S), 2)
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
