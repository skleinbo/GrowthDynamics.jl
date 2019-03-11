module TumorObservables

import IndexedTables: table, join, renamecol

import OpenCLPicker: @opencl

@opencl import OpenCL
import MetaGraphs: nv, inneighbors, neighborhood, neighborhood_dists, set_prop!, get_prop, vertices


import ..Lattices
import ..LatticeTumorDynamics

import ..TumorConfigurations: TumorConfiguration

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
        has_children

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
    ps_table = table(population_size(state.lattice, 0), pkey=[1])
    ps_table = renamecol(ps_table, 1=>:g, 2=>:npop)

    # verts = getindex.(nb,1)
    join(ps_table, nb_table)
end

## SLOW, to many passes over the whole graph.
## Should be doable in one pass!
function cphylo_hist(state::TumorConfiguration)
    P = state.Phylogeny
    nb = neighborhood_dists(P, 1, nv(P), dir=:in)
    ps_table = table(population_size(state.lattice, 0), pkey=[1])
    nb_table = table(map(nb) do x
        (get_prop(P, x[1], :genotype),x[1], x[2])
    end, pkey=[1])
    nb_table = renamecol(nb_table, 1=>:g, 2=>:nv, 3=>:dist)
    ps_table = renamecol(ps_table, 1=>:g, 2=>:npop)

    joined = join(nb_table, ps_table, how=:outer)

    map(vertices(P)) do bv
        nb = neighborhood(P, bv, nv(P), dir=:in)
        (nv=bv, dist=filter(x->x.nv==bv, joined)[1].dist,
        cnpop=mapreduce(+, nb) do n
            x = filter(x->x.nv==n, joined)[1].npop
            if ismissing(x)
                return 0.0
            else
                x
            end
        end)
    end
end

##end module
end
