module LatticeTumorDynamics

export  moran!,
        independent_death_birth!,
        die_or_proliferate!

import MetaGraphs: nv, add_vertex!, add_edge!, set_indexing_prop!, set_prop!, get_prop
using StatsBase: Weights,sample
import Random: shuffle!

using ..Lattices
import ..TumorConfigurations

# using OffLattice


occupied(m,n,s,N) = @inbounds m<1||m>N||n<1||n>N || s[x,y]!=0
growth_rate(nw,basebr) = basebr*(1 - 1/6*nw)



## Weird method to generate rand(1:N)
## ~3x speedup
@inline rand1toN(N) = rand(1:N)

###--- Start of simulation methods ---###

function moran!(
    lattice::LT,
    fitness::Vector{Float64};
    T=0,
    mu::Float64=10.0^-2,
    f_mut=(L,G,g)->maximum(LightGraphs.vertices(G))+1,
    constraint=true) where {LT<:AbstractLattice{<:Integer}}

    genealogy = lattice.Phylogeny

    I = CartesianIndices(lattice.data)
    Lin = LineearIndices(lattice.data)

    dim = length(size(lattice.data))
    lin_N = size(lattice.data,1)
    tot_N = lin_N^dim

    genotype = maximum(LightGraphs.vertices(genealogy))

    fitness_lattice = vec([k!=0 ? fitness[k] : 0. for k in lattice.data])

    nn = neighbours(lattice, CartesianIndex{dim}()) # Initialize neighbours to the appr. type
    new = 0
    old = 0
    selected = 0
    for t in 1:T
        ## Pick one to proliferate
        old = sample(1:tot_N, Weights(fitness_lattice))

        if !constraint
            new = rand(1:tot_N)
            while new == old
                new = rand(1:tot_N)
            end
        else
            neighbours!(nn,I[old],lattice)
            nz_nn = filter(x->!out_of_bounds(x, lin_N), nn)
            selected = rand(nz_nn)
            new = Lin[selected]
        end

        genotype = f_mut(lattice,genealogy,lattice.data[old])
        if rand()<mu && genotype<=length(fitness)
            # LightGraphs.add_vertex!(genealogy)
            # LightGraphs.add_edge!(genealogy,lattice.data[old],genotype)

            lattice.data[new] = genotype
            fitness_lattice[new] = fitness[genotype]
        else
            lattice.data[new] = lattice.data[old]
            fitness_lattice[new] = fitness_lattice[old]
        end
    end
end

function density!(nn,L,ind::CartesianIndex)
    lin_N = size(L.data,1)
    neighbours!(nn, ind, L)
    tot = count(x->!out_of_bounds(x,lin_N),nn) #count(x->!out_of_bounds(x...,lin_N), nn)
    nz =  count(x->!out_of_bounds(x,lin_N) && L.data[x]!=0, nn)
    return nz/tot
end


function die_or_proliferate!(
    ;state=TumorConfigurations.uniform_circle(0),
    fitness=()->0.0,
    T=0,
    mu::Float64=0.0,
    f_mut=(L,G,g)->maximum(LightGraphs.vertices(G))+1,
    d::Float64=1/100,
    constraint=true,
    DEBUG=false,
    callback=s->begin end,
    abort=s->false,
    kwargs...)

    phylogeny = state.Phylogeny

    I = CartesianIndices(state.lattice.data)
    Lin = LinearIndices(state.lattice.data)

    dim = length(size(state.lattice.data))
    lin_N = size(state.lattice.data,1)
    tot_N = length(state.lattice.data)

    fitness_lattice = vec([k!=0 ? fitness(k) : 0. for k in state.lattice.data])
    br_lattice = zeros(tot_N)

    nonzeros = count(x->x!=0, state.lattice.data)
    base_br = 1.0 - d

    nn = neighbours(state.lattice, CartesianIndex{dim}()) # Initialize neighbours to the appr. type
    neighbour_indices = collect(1:length(nn))

    for k in 1:tot_N
        br_lattice[k] = (1.0-density!(nn,state.lattice,I[k])) * base_br * fitness_lattice[k]
    end

    new = 0
    new_cart = nn[1]
    old = 0
    selected = 0
    br = 0.
    cumrate = 0.
    validneighbour = false
    action = :none
    total_rate = mapreduce(+, enumerate(state.lattice.data)) do x x[2]>0 ? d + br_lattice[x[1]] : 0. end
    DEBUG && println(total_rate)

    @inbounds for t in 0:T
        Base.invokelatest(callback,state)
        if abort(state)
            break
        end
        ## Much cheaper than checking the whole lattice each iteration
        ## Leave the loop if lattice is empty
        if nonzeros==0
            break
        end

        ## Die, proliferate or be dormant
        who_and_what = rand()*total_rate

        cumrate = 0.
        selected = 0
        action = :none
        while selected < tot_N
            selected += 1
            if state[selected] != 0
                cumrate += d
                if cumrate > who_and_what
                    action = :die
                    break
                end
                cumrate += br_lattice[selected]
                if cumrate > who_and_what
                    action = :proliferate
                    break
                end
            end
        end

        DEBUG && println(br_lattice)
        if action == :die
            DEBUG && println("Die")
            nonzeros -= 1
            total_rate -= br_lattice[selected] + d

            g = state.lattice.data[selected]
            old_ps = get_prop(phylogeny, phylogeny[g, :genotype], :npop)
            set_prop!(phylogeny, phylogeny[g, :genotype], :npop, min(0,old_ps-1))

            state[selected] = 0
            fitness_lattice[selected] = 0.
            br_lattice[selected] = 0.
            ## Update birth-rates
            neighbours!(nn, I[selected], state.lattice)
            for n in nn
                if !out_of_bounds(n, lin_N)
                    j = Lin[n]
                    # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                    if state[j] != 0
                        DEBUG && println("adjusting rates on $j by $(fitness_lattice[j])")
                        total_rate -= br_lattice[j]
                        br_lattice[j] +=  1.0/length(nn) * fitness_lattice[j] * base_br
                        total_rate += br_lattice[j]
                    end
                end
            end
            ##
        elseif action == :proliferate
            DEBUG && println("Live")
            old = selected
            if !constraint
                new = 0
                while new != old
                    new = rand1toN(tot_N)
                end
            else
                neighbours!(nn, I[old], state.lattice)
                validneighbour = false
                for j in shuffle!(neighbour_indices)
                    if !out_of_bounds(nn[j],lin_N) && state[nn[j]]==0
                        new_cart = nn[j]
                        validneighbour = true
                        break
                    end
                end
                if !validneighbour
                    continue
                end

                new = Lin[new_cart]
                ## Mutation
                # @assert genotype!=0
                genotype = state[old]
                if rand()<mu
                    new_genotype = f_mut(state, phylogeny, genotype)
                    if new_genotype != genotype && fitness(new_genotype)!=-Inf # -Inf indicates no mutation possible
                        if !in(new_genotype, keys(phylogeny.metaindex[:genotype]))
                            add_vertex!(phylogeny)
                            set_indexing_prop!(phylogeny, nv(phylogeny), :genotype, new_genotype)
                            set_prop!(phylogeny, nv(phylogeny), :T, state.t)
                            set_prop!(phylogeny, nv(phylogeny), :npop, 0)
                        end
                        parent_vertex = phylogeny.metaindex[:genotype][genotype]
                        add_edge!(phylogeny,nv(phylogeny),parent_vertex)

                        genotype = new_genotype
                    end
                end

                g = state.lattice.data[selected]
                old_ps = get_prop(phylogeny, phylogeny[g, :genotype], :npop)
                set_prop!(phylogeny, phylogeny[g, :genotype], :npop, old_ps+1)

                state[new] = genotype
                fitness_lattice[new] = fitness(genotype)

                br_lattice[new] = (1.0-density!(nn,state.lattice,I[new])) * base_br * fitness_lattice[new]
                nonzeros += 1
                total_rate += d + br_lattice[new]

                neighbours!(nn, I[new], state.lattice)
                for n in nn
                    if !out_of_bounds(n, lin_N)
                        j = Lin[n]
                        # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                        if state[j] != 0
                            total_rate -= br_lattice[j]
                            br_lattice[j] -=  1.0/length(nn) * fitness_lattice[j] * base_br
                            total_rate += br_lattice[j]
                        end
                    end
                end
            end
        else
            DEBUG && println("Noone")
        end
        state.t += 1
        state.treal += -1/total_rate*log(1-rand())
    end
    # @assert (mapreduce(+, enumerate(state.lattice.data)) do x x[2]>0 ? d + br_lattice[x[1]] : 0.; end) ≈ total_rate
end

function independent_death_birth!(
    lattice::LT,
    fitness;
    T=0,
    mu::Float64=10.0^-2,
    f_mut=(L,G,g)->maximum(LightGraphs.vertices(G))+1,
    d::Float64=1/100,
    constraint=true) where LT<:AbstractLattice{<:Integer}

    genealogy::lattice.Phylogeny

    I = CartesianIndices(state.lattice.data)
    Lin = LinearIndices(state.lattice)

    dim = length(size(lattice.data))
    lin_N = size(lattice.data,1)
    tot_N = length(lattice.data)

    fitness_lattice = vec([k!=0 ? fitness[k] : 0. for k in lattice.data])
    br_lattice = zeros(tot_N)


    nn = neighbours(lattice, CartesianIndex{dim}()) # Initialize neighbours to the appr. type

    for k in 1:tot_N
        br_lattice[k] = max(0., (1.0-density!(nn,lattice,k)) * 1. * fitness_lattice[k])
    end
    # @assert iszero(br_lattice)

    new = 0
    old = 0
    selected = 0
    br = 0.
    for t in 1:T
        ## Pick one to possibly die
        if rand() < d && !iszero(lattice.data)
            i = rand(1:tot_N)
            while lattice.data[i] == 0
                i = rand(1:tot_N)
            end
            lattice.data[i] = 0
            fitness_lattice[i] = 0.
            br_lattice[i] = 0.
            ## Update birth-rates
            neighbours!(nn, I[i], lattice)
            for n in nn
                if !out_of_bounds(n, lin_N)
                    j = Lin[n]
                    # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                    br_lattice[j] +=  1.0/length(nn) * fitness_lattice[j]
                end
            end
            ##
        end

        ## Pick one to proliferate
        if iszero(br_lattice)
            # println("death without birth")
            continue
        else
            old = sample(1:tot_N, Weights(br_lattice))
        end
        # if br_lattice[old] == 0.
        #     println(t, "\t", old,"\t", lattice.data[old],"\t", fitness_lattice[old])
        #     return (lattice, br_lattice, fitness_lattice)
        # end
        if !constraint
            new = 0
            while new != old
                new = rand1toN(tot_N)
            end
        else
            neighbours!(nn, I[old], lattice)
            z_nn = filter(x->!out_of_bounds(x,lin_N) && lattice.data[x]==0, nn)
            if isempty(z_nn)
                continue
            end
            selected = rand(z_nn)
            new = Lin[selected]
            ## Mutation
            genotype = f_mut(lattice,genealogy,lattice.data[old])
            # @assert genotype!=0
            if rand()<mu && genotype<=length(fitness)
                LightGraphs.add_vertex!(genealogy)
                LightGraphs.add_edge!(genealogy,lattice.data[old],genotype)
                lattice.data[new] = genotype
                fitness_lattice[new] = fitness[genotype]
            else
                lattice.data[new] = lattice.data[old]
                fitness_lattice[new] = fitness_lattice[old]
            end

            br_lattice[new] = max(0., (1.0-density!(nn,lattice,new)) * 1. * fitness_lattice[new] )
            neighbours!(nn, I[new], lattice)
            for n in nn
                if !out_of_bounds(n, lin_N)
                    j = Lin[n]
                    # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                    br_lattice[j] +=  1.0/length(nn) * fitness_lattice[j]
                end
            end
        end


    end
end

dynamics_dict = Dict(
    :moran => moran!,
    :die_or_proliferate => die_or_proliferate!,
    :indy_birth_death => independent_death_birth!
)

## -- END module LatticeTumorDynamics --
end
