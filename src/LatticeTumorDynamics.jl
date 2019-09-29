module LatticeTumorDynamics

export  prune_me!,
        moran!,
        independent_death_birth!,
        die_or_proliferate!

import LightGraphs: nv, vertices, add_vertex!, add_edge!

using StatsBase: Weights,sample,mean
import Distributions: Binomial, Exponential, cdf

import Random: shuffle!

using ..Lattices
import ..TumorConfigurations: TumorConfiguration

import ..Phylogenies: annotate_snps!, prune_phylogeny!

# using OffLattice


occupied(m,n,s,N) = @inbounds m<1||m>N||n<1||n>N || s[x,y]!=0
growth_rate(nw,basebr) = basebr*(1 - 1/6*nw)



## Weird method to generate rand(1:N)
## ~3x speedup
@inline rand1toN(N) = rand(1:N)

const MutationProfile = Tuple{Symbol, Float64, Int64} # (rate, :poisson/:fixed, genome length)

###--- Start of simulation methods ---###

"""
    exponential!(state::NoLattice{Int}; <keyword arguments>)

Run exponential growth on an unstructered population.

# Arguments
- `T::Int`: the number of steps (generations) to advance.
- `fitness`: function that assigns a fitness value to a genotype `g::Int`.
- `mu`: mutation rate.
- `baserate`: progressing real time is measured in `1/baserate`.
- `prune_period`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function exponential!(
    state::TumorConfiguration{NoLattice{Int64}};
    fitness=g->1.0,
    T=0,
    mu::Float64=0.0,
    d::Float64=0.0,
    baserate=1.0,
    prune_period=0,
    prune_on_exit=true,
    DEBUG=false,
    callback=(s,t)->begin end,
    abort=s->false,
    kwargs...)

    # P = state.Phylogeny
    K = state.lattice.N # Carrying capacity

    genotypes = state.meta.genotypes
    npops = state.meta.npops
    fitnesses = state.meta.fitnesses
    rates = fitnesses.*npops
    snps = state.meta.snps

    Ntotal = sum(npops)
    total_rate = sum(rates)

    if haskey(kwargs, :det_mutations) && kwargs[:det_mutations] == true
        p_mu = 1.0
    else
        p_mu = 1.0 - exp(-mu)
    end

    wrates = Weights(rates)
    wnpops = Weights(npops)
    new = 0
    old = 0
    selected = 0

    for step in 0:T
        if prune_period > 0 && state.t > 0 && (state.t)%prune_period==0
            @debug "Pruning..."
            prune_me!(state, mu)
        end

        Base.invokelatest(callback,state,state.t)
        if abort(state)
            break
        end
        # In case we pruned, renew bindings
        genotypes = state.meta.genotypes
        npops = state.meta.npops
        fitnesses = state.meta.fitnesses
        rates = fitnesses.*npops
        snps = state.meta.snps
        wrates = Weights(rates)
        wnpops = Weights(npops)

        # @debug "Step $step/$T"
        told = state.treal
        state.treal += 1.0/(baserate*mean(fitnesses))
        dt = state.treal - told

        # If carrying capacity is reached, we exit.
        if K <= Ntotal
            break
        end

        for (j,genotype) in enumerate(genotypes)|>collect
            # @debug "Genotype: $j,$genotype"
            if npops[j] == 0
                # @debug "g$genotype is empty; skipping"
                continue
            end
            if haskey(kwargs, :det_growth) && kwargs[:det_growth] == true
                pgrow = 1.0 # grow with certainty
                nplus = min(K-Ntotal, ceil(Int, npops[j]*baserate))
            else
                pgrow = 1.0 - exp(-fitnesses[j]/mean(fitnesses)) #CDF of exponential with s/<s>
                nplus = min(rand(Binomial(npops[j], pgrow)), K-Ntotal) # How many proliferate?
            end
            # @debug "Pgrow = $pgrow"
            if nplus<=0
                continue
            end
            nmutants = rand(Binomial(nplus, p_mu)) # How many of those mutate?
            for _ in 1:nmutants
                new_genotype = genotypes[end] + 1
                if new_genotype != genotype && fitness(new_genotype)!=-Inf # -Inf indicates no mutation possible
                    if true || !in(new_genotype, genotypes)
                        push!(state, new_genotype)
                        fitnesses[end] = fitness(new_genotype)
                        push!(rates, 0.0)
                    end
                    add_edge!(state.Phylogeny,nv(state.Phylogeny),j)
                    new = length(genotypes)
                    rates[new] += fitnesses[new]
                    total_rate += fitnesses[new]
                    npops[new] += 1
                end
            end
            npops[j] += nplus - nmutants
            Ntotal += nplus
        end

        state.t += 1
    end
    if prune_on_exit
        prune_me!(state, mu)
    end
    # state.Phylogeny = P
    nothing
end

function prune_me!(state,mu)
    annotate_snps!(state, mu)
    prune_phylogeny!(state)
    # state.Phylogeny = newP
    # state.meta = newMeta
    nothing
end
_prune!(s) = LatticeTumorDynamics.prune_me!(s, mu)

"""
    moran!(state::NoLattice{Int}; <keyword arguments>)

(Extended) Moran dynamics on an unstructured population. Grow until carrying capacity
is reach. After that individuals begin replacing each other.

# Arguments
- `T::Int`: the number of steps to advance.
- `fitness`: function that assigns a fitness value to a genotype `g::Int`.
- `mu`: mutation rate.
- `d`: death rate.
- `baserate`: progressing real time is measured in `1/baserate`.
- `prune_period`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function moran!(
    state::TumorConfiguration{NoLattice{Int64}};
    fitness=g->1.0,
    T=0,
    mu::Float64=0.0,
    d::Float64=0.0,
    prune_period=0,
    prune_on_exit=true,
    DEBUG=false,
    callback=(s,t)->begin end,
    abort=s->false,
    kwargs...)

    K = state.lattice.N # Carrying capacity

    genotypes = state.meta.genotypes
    npops = state.meta.npops
    fitnesses = state.meta.fitnesses
    rates = fitnesses.*npops
    snps = state.meta.snps

    Ntotal = sum(npops)
    total_rate = sum(rates)

    p_mu = 1.0 - exp(-mu)

    wrates = Weights(rates)
    wnpops = Weights(npops)
    new = 0
    old = 0
    selected = 0

    for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t)%prune_period==0
            @debug "Pruning..."
            prune_me!(state, mu)
        end

        Base.invokelatest(callback,state,state.t)
        if abort(state)
            break
        end
        # In case we pruned, renew bindings
        genotypes = state.meta.genotypes
        npops = state.meta.npops
        fitnesses = state.meta.fitnesses
        rates = fitnesses.*npops
        snps = state.meta.snps
        wrates = Weights(rates)
        wnpops = Weights(npops)
        ## Pick one to proliferate
        old = sample(wrates)
        new = sample(wnpops)
        # @debug old, new
        genotype = genotypes[old]
        if rand()<p_mu
            new_genotype = maximum(genotypes)+1
            if new_genotype != genotype && fitness(new_genotype)!=-Inf # -Inf indicates no mutation possible
                if true || !in(new_genotype, genotypes)
                    push!(state, new_genotype)
                    fitnesses[end] = fitness(new_genotype)
                    push!(rates, 0.0)
                end
                add_edge!(state.Phylogeny,nv(state.Phylogeny),old)
                old = length(genotypes)
            end
        end
        rates[old] += fitnesses[old]
        total_rate += fitnesses[old]
        npops[old] += 1
        Ntotal += 1

        # If carrying capacity is reached, one needs to die.
        if K < Ntotal
            rates[new] -= fitnesses[new]
            total_rate += fitnesses[new]
            npops[new] -= 1
            Ntotal -= 1
            ## If the carrying capacity is reached, the process is
            ## d-limited
            state.treal += -1.0/(Ntotal*d)*log(1.0-rand())
            state.treal += -1.0/total_rate*log(1.0-rand())
        else
            state.treal += -1.0/total_rate*log(1.0-rand())
        end


        state.t += 1
    end
    if prune_on_exit
        prune_me!(state, mu)
    end
end

## Define density function for different lattice types.
for LatticeType in [Lattices.LineLattice, Lattices.HexagonalLattice]
    if LatticeType <: Lattices.HexagonalLattice
        nn_function = :hex_nneighbors
    elseif LatticeType <: Lattices.LineLattice
        nn_function = :line_nneighbors
    end

    eval(
    quote
        function density!(nn,L::$LatticeType,ind::CartesianIndex)
            lin_N = size(L.data, 1)
            neighbors!(nn, ind, L)
            tot = $(nn_function)(ind,lin_N)
            nz =  count(x->!out_of_bounds(x,lin_N) && L.data[x]!=0, nn)
            return nz/tot
        end
    end)
end

"""
    die_or_proliferate!(state::RealLattice{Int}; <keyword arguments>)

Moran-like dynamics on an spatially structured population. Each step is either a death or
(potential) birth and mutation event.

Individuals die at a rate `d`.
Birthrate depends linearily on the number of neighbors.

# Arguments
- `T::Int`: the number of steps to advance.
- `fitness`: function that assigns a fitness value to a genotype `g::Int`.
- `mu`: mutation rate.
- `d`: death rate. Zero halts the dynamics after carrying capacity is reached.
- `baserate`: progressing real time is measured in `1/baserate`.
- `prune_period`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function die_or_proliferate!(
    state::TumorConfiguration{<:RealLattice};
    fitness=g->0.0,
    T=0,
    mu::Float64=0.0,
    d::Float64=1/100,
    constraint=true,
    prune_period=0,
    prune_on_exit=true,
    DEBUG=false,
    callback=(s,t)->begin end,
    abort=s->false,
    kwargs...)

    genotypes = state.meta.genotypes
    npops = state.meta.npops
    fitnesses = state.meta.fitnesses
    snps = state.meta.snps

    I = CartesianIndices(state.lattice.data)
    Lin = LinearIndices(state.lattice.data)

    dim = length(size(state.lattice.data))
    lin_N = size(state.lattice.data,1)
    tot_N = length(state.lattice.data)

    fitness_lattice = vec([k!=0 ? fitnesses[findfirst(x->x==k, genotypes)] : 0. for k in state.lattice.data])
    br_lattice = zeros(tot_N)

    nonzeros = count(x->x!=0, state.lattice.data)
    base_br = 1.0 - d

    nn = neighbors(state.lattice, CartesianIndex{dim}()) # Initialize neighbors to the appr. type
    neighbor_indices = collect(1:length(nn))

    for k in 1:tot_N
        br_lattice[k] = (1.0-density!(nn,state.lattice,I[k])) * base_br * fitness_lattice[k]
    end

    p_mu = 1.0 - exp(-mu)

    new = 0
    new_cart = nn[1]
    old = 0
    selected = 0
    br = 0.
    cumrate = 0.
    validneighbor = false
    action = :none
    total_rate = mapreduce(+, enumerate(state.lattice.data)) do x x[2]>0 ? d + br_lattice[x[1]] : 0. end
    @debug total_rate

    @debug "Prune period is $prune_period"
    @inbounds for t in 0:T
        #prune_me!()
        #@debug "t=$(state.t)"
        if prune_period > 0 && state.t > 0 && (state.t)%prune_period==0
            #@debug "Pruning..."
            prune_me!(state, mu)
        end
        Base.invokelatest(callback,state,state.t)
        if abort(state)
            break
        end
        # In case we pruned, renew bindings.
        genotypes = state.meta.genotypes
        npops = state.meta.npops
        fitnesses = state.meta.fitnesses
        snps = state.meta.snps
        ## Much cheaper than checking the whole lattice each iteration
        ## Leave the loop if lattice is empty
        if nonzeros==0
            @info "Lattice empty. Exiting."
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

        #@debug br_lattice
        if action == :die
            #@debug "Die"
            nonzeros -= 1
            total_rate -= br_lattice[selected] + d

            g = state.lattice.data[selected]
            g_id = findfirst(x->x==g, genotypes)
            old_ps = npops[g_id]
            npops[g_id] = max(0,old_ps-1)

            state[selected] = 0
            fitness_lattice[selected] = 0.
            br_lattice[selected] = 0.
            ## Update birth-rates
            neighbors!(nn, I[selected], state.lattice)
            for n in nn
                if !out_of_bounds(n, lin_N) && state[n]!=0
                    j = Lin[n]
                    # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                    #@debug "adjusting rates on $j by $(fitness_lattice[j])"
                    total_rate -= br_lattice[j]
                    br_lattice[j] +=  1.0/nneighbors(state.lattice, n, lin_N) * fitness_lattice[j] * base_br
                    total_rate += br_lattice[j]
                end
            end
            ##
        elseif action == :proliferate
            #@debug "Live"
            old = selected
            if !constraint
                new = 0
                while new != old
                    new = rand1toN(tot_N)
                end
            else
                neighbors!(nn, I[old], state.lattice)
                validneighbor = false
                for j in shuffle!(neighbor_indices)
                    if !out_of_bounds(nn[j],lin_N) && state[nn[j]]==0
                        new_cart = nn[j]
                        validneighbor = true
                        break
                    end
                end
                if !validneighbor
                    continue
                end

                new = Lin[new_cart]
                ## Mutation
                # @assert genotype!=0
                genotype = state[old]
                g_id = findfirst(x->x==genotype, genotypes)
                if rand()<p_mu
                    new_genotype = maximum(genotypes)+1
                    if new_genotype != genotype && fitness(new_genotype)!=-Inf # -Inf indicates no mutation possible
                        if true || !in(new_genotype, keys(phylogeny.metaindex[:genotype]))
                            push!(state, new_genotype)
                            fitnesses[end] = fitness(new_genotype)
                        end
                        add_edge!(state.Phylogeny,nv(state.Phylogeny),g_id)
                        genotype = new_genotype
                        g_id = length(genotypes)
                    end
                end

                state[new] = genotype
                npops[g_id] += 1
                fitness_lattice[new] = fitnesses[g_id]

                br_lattice[new] = (1.0-density!(nn,state.lattice,I[new])) * base_br * fitness_lattice[new]
                nonzeros += 1
                total_rate += d + br_lattice[new]

                neighbors!(nn, I[new], state.lattice)
                for n in nn
                    if !out_of_bounds(n, lin_N) && state[n]!=0
                        j = Lin[n]
                        # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
                        total_rate -= br_lattice[j]
                        br_lattice[j] -=  1.0/nneighbors(state.lattice, n,lin_N) * fitness_lattice[j] * base_br
                        total_rate += br_lattice[j]
                    end
                end
            end
        else
            @debug "Noone"
        end
        state.t += 1
        state.treal += -1.0/total_rate*log(1.0-rand())

        # global flattice = fitness_lattice
        # global brlattice = br_lattice
        # global mystate = state
    end
    ## Update the phylogeny
    if prune_on_exit
        prune_me!(state, mu)
    end
    @debug "Done at $(state.t)"
    # @assert (mapreduce(+, enumerate(state.lattice.data)) do x x[2]>0 ? d + br_lattice[x[1]] : 0.; end) ≈ total_rate
end

# function independent_death_birth!(
#     lattice::LT,
#     fitness;
#     T=0,
#     mu::Float64=10.0^-2,
#     f_mut=(L,G,g)->maximum(LightGraphs.vertices(G))+1,
#     d::Float64=1/100,
#     constraint=true) where LT<:RealLattice{<:Integer}
#
#     genealogy::lattice.Phylogeny
#
#     I = CartesianIndices(state.lattice.data)
#     Lin = LinearIndices(state.lattice)
#
#     dim = length(size(lattice.data))
#     lin_N = size(lattice.data,1)
#     tot_N = length(lattice.data)
#
#     fitness_lattice = vec([k!=0 ? fitness[k] : 0. for k in lattice.data])
#     br_lattice = zeros(tot_N)
#
#
#     nn = neighbors(lattice, CartesianIndex{dim}()) # Initialize neighbors to the appr. type
#
#     for k in 1:tot_N
#         br_lattice[k] = max(0., (1.0-density!(nn,lattice,k)) * 1. * fitness_lattice[k])
#     end
#     # @assert iszero(br_lattice)
#
#     new = 0
#     old = 0
#     selected = 0
#     br = 0.
#     for t in 1:T
#         ## Pick one to possibly die
#         if rand() < d && !iszero(lattice.data)
#             i = rand(1:tot_N)
#             while lattice.data[i] == 0
#                 i = rand(1:tot_N)
#             end
#             lattice.data[i] = 0
#             fitness_lattice[i] = 0.
#             br_lattice[i] = 0.
#             ## Update birth-rates
#             neighbors!(nn, I[i], lattice)
#             for n in nn
#                 if !out_of_bounds(n, lin_N)
#                     j = Lin[n]
#                     # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
#                     br_lattice[j] +=  1.0/length(nn) * fitness_lattice[j]
#                 end
#             end
#             ##
#         end
#
#         ## Pick one to proliferate
#         if iszero(br_lattice)
#             # println("death without birth")
#             continue
#         else
#             old = sample(1:tot_N, Weights(br_lattice))
#         end
#         # if br_lattice[old] == 0.
#         #     println(t, "\t", old,"\t", lattice.data[old],"\t", fitness_lattice[old])
#         #     return (lattice, br_lattice, fitness_lattice)
#         # end
#         if !constraint
#             new = 0
#             while new != old
#                 new = rand1toN(tot_N)
#             end
#         else
#             neighbors!(nn, I[old], lattice)
#             z_nn = filter(x->!out_of_bounds(x,lin_N) && lattice.data[x]==0, nn)
#             if isempty(z_nn)
#                 continue
#             end
#             selected = rand(z_nn)
#             new = Lin[selected]
#             ## Mutation
#             genotype = f_mut(lattice,genealogy,lattice.data[old])
#             # @assert genotype!=0
#             if rand()<mu && genotype<=length(fitness)
#                 LightGraphs.add_vertex!(genealogy)
#                 LightGraphs.add_edge!(genealogy,lattice.data[old],genotype)
#                 lattice.data[new] = genotype
#                 fitness_lattice[new] = fitness[genotype]
#             else
#                 lattice.data[new] = lattice.data[old]
#                 fitness_lattice[new] = fitness_lattice[old]
#             end
#
#             br_lattice[new] = max(0., (1.0-density!(nn,lattice,new)) * 1. * fitness_lattice[new] )
#             neighbors!(nn, I[new], lattice)
#             for n in nn
#                 if !out_of_bounds(n, lin_N)
#                     j = Lin[n]
#                     # br_lattice[j] = max(0., (1.-density(lattice,j)) * 1. * fitness_lattice[j] )
#                     br_lattice[j] +=  1.0/length(nn) * fitness_lattice[j]
#                 end
#             end
#         end
#
#
#     end
# end

dynamics_dict = Dict(
    :moran => moran!,
    :die_or_proliferate => die_or_proliferate!,
)

## -- END module LatticeTumorDynamics --
end
