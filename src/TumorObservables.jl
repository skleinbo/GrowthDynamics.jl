module TumorObservables

import OpenCLPicker: @opencl

@opencl import OpenCL
import MetaGraphs: inneighbors


import ..Lattices
import ..LatticeTumorDynamics

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

function allelic_fractions(L::Lattices.AbstractLattice{<:Integer})
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

function total_population_size(L::Lattices.AbstractLattice{<:Integer})
    countnz(L.data)
end

function population_size(L::Lattices.AbstractLattice{<:Integer}, t)
    m = maximum(L.data)
    if m == 0
        return []
    end
    genotypes = 1:m
    Ng = [(g,0.) for g in genotypes]
    for g in genotypes
        Ng[g] = (g,count(x->x==g,L.data))
    end
    return Ng
end

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

function surface(L::Lattices.AbstractLattice{<:Integer}, g::Int)
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
function surface2(L::Lattices.AbstractLattice{<:Integer}, g::Int)
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

function nchildren(L,g)
    vertex = L.Phylogeny[:genotype][g]
    length(inneighbors(L.Phylogeny, vertex))
end

has_children(L,g) = nchildren(L,g) > 0

##end module
end
