module LatticeTumorDynamics

export prune_me!,
        moran!,
        independent_death_birth!,
        die_or_proliferate!,
        twonew!

import Graphs: nv, vertices, add_vertex!, add_edge!

using StatsBase: Weights,sample,mean
import Distributions: Binomial, Exponential, cdf

import Random: shuffle!

using ..Lattices
import ..TumorConfigurations: TumorConfiguration, getfitness, gindex, _resize!

import ..Phylogenies: annotate_snps!, add_snps!, prune_phylogeny!, sample_ztp

import ..TumorObservables: total_population_size

occupied(m,n,s,N) = @inbounds m < 1 || m > N || n < 1 || n > N || s[x,y] != 0
growth_rate(nw,basebr) = basebr * (1 - 1 / 6 * nw)

@enum Action none=0 proliferate=1 mutate=2 die=3


## Weird method to generate rand(1:N)
## ~3x speedup
@inline rand1toN(N) = rand(1:N)

const MutationProfile = Tuple{Symbol,Float64,Int64} # (rate, :poisson/:fixed, genome length)

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
    state::TumorConfiguration{Int64, NoLattice{Int64}};
    fitness = g -> 1.0,
    T = 0,
    mu::Float64 = 0.0,
    d::Float64 = 0.0,
    baserate = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    DEBUG = false,
    callback = (s, t) -> begin end,
    abort = s -> false,
    kwargs...)

    # P = state.phylogeny
    K = state.lattice.N # Carrying capacity

    genotypes = state.meta.genotypes
    npops = state.meta.npops
    fitnesses = state.meta.fitnesses
    rates = fitnesses .* npops
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
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            @debug "Pruning..."
            prune_me!(state, mu)
        end

        Base.invokelatest(callback, state, state.t)
        if abort(state)
            break
        end
        # In case we pruned, renew bindings
        genotypes = state.meta.genotypes
        npops = state.meta.npops
        fitnesses = state.meta.fitnesses
        rates = fitnesses .* npops
        snps = state.meta.snps
        wrates = Weights(rates)
        wnpops = Weights(npops)

        # @debug "Step $step/$T"
        told = state.treal
        state.treal += 1.0 / (baserate * mean(fitnesses))
        dt = state.treal - told

        # If carrying capacity is reached, we exit.
        if K <= Ntotal
            break
        end

        for (j, genotype) in enumerate(genotypes) |> collect
            # @debug "Genotype: $j,$genotype"
            if npops[j] == 0
                # @debug "g$genotype is empty; skipping"
                continue
            end
            if haskey(kwargs, :det_growth) && kwargs[:det_growth] == true
                pgrow = 1.0 # grow with certainty
                nplus = min(K - Ntotal, ceil(Int, npops[j] * baserate))
            else
                pgrow = 1.0 - exp(-fitnesses[j] / mean(fitnesses)) # CDF of exponential with s/<s>
                nplus = min(rand(Binomial(npops[j], pgrow)), K - Ntotal) # How many proliferate?
            end
            # @debug "Pgrow = $pgrow"
            if nplus <= 0
                continue
            end
            nmutants = rand(Binomial(nplus, p_mu)) # How many of those mutate?
            for _ in 1:nmutants
                new_genotype = genotypes[end] + 1
                if new_genotype != genotype && fitness(new_genotype) != -Inf # -Inf indicates no mutation possible
                    if true || !in(new_genotype, genotypes)
                        push!(state, new_genotype)
                        fitnesses[end] = fitness(new_genotype)
                        push!(rates, 0.0)
                    end
                    add_edge!(state.phylogeny, nv(state.phylogeny), j)
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
    # state.phylogeny = P
    nothing
end

function prune_me!(state, mu)
    annotate_snps!(state, mu)
    prune_phylogeny!(state)
    # state.phylogeny = newP
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
- `p_grow=1.0`: Probability with which to actually proliferate. If no proliferation happens, mutation might still occur.
- `mu`: mutation rate.
- `mu_type=[:poisson, :fixed]`: Number of mutations is fixed, or Poisson-distributed.
- `genome_length=10^9`: Length of the haploid genome.
- `d`: death rate.
- `baserate`: progressing real time is measured in `1/baserate`.
- `prune_period`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function moran!(
    state::TumorConfiguration{Int64, NoLattice{Int64}};
    fitness = (s, gold, gnew) -> 1.0,
    T = 0,
    mu::Float64 = 0.0,
    mu_type = :poisson,
    makesnps = true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    d::Float64 = 0.0,
    p_grow = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    DEBUG = false,
    callback = (s, t) -> begin end,
    abort = s -> false,
    K::Int64 = typemax(Int64), # carrying capacity
    sizehint=0, # initially resize state.meta to `sizehint`. do nothing if 0.
    kwargs...)

    # genotypes = state.meta.genotypes
    # npops = state.meta.npops
    # fitnesses = state.meta.fitnesses
    rates = state.meta[:, :fitnesses] .* state.meta[:, :npops]
    # snps = state.meta.snps

    ## sizehint ##
    if sizehint > length(state.meta.genotypes)
        _resize!(state.meta, sizehint)
    end

    Ntotal = total_population_size(state)
    total_rate = sum(rates) + d*Ntotal

    p_mu = 1.0 - exp(-mu)


    new = 0
    old = 0
    selected = 0

    @inbounds for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            @debug "Pruning..."
            prune_me!(state, mu)
            # # Renew bindings
            # genotypes = state.meta.genotypes
            # npops = state.meta.npops
            # fitnesses = state.meta.fitnesses
            # rates = fitnesses .* npops
            # snps = state.meta.snps
            # wrates = Weights(rates)
            # wnpops = Weights(npops)
        end

        callback(state, state.t)
        if abort(state)
            break
        end

        wrates = Weights(rates, total_rate - d*Ntotal)
        wnpops = Weights((@view state.meta.npops[begin:state.meta._len]), Ntotal)

        b_die = rand() < d / (total_rate/Ntotal)

        if b_die
            die = sample(wnpops)
            rates[die] -= state.meta[die, Val(:fitness)]
            total_rate -= state.meta[die, Val(:fitness)] + d
            state.meta[die, Val(:npop)] -= 1
            Ntotal -= 1
        else
            ## Pick one to proliferate
            old = sample(wrates)
            # rate = 0.0
            # target = rand()*total_rate
            # old = 0
            # while rate < target
            #     old += 1
            #     @inbounds rate += rates[old]
            # end

            b_grow = p_grow==1.0 || rand() < p_grow
            if !b_grow
                rates[old] -= state.meta[old, Val(:fitness)]
                total_rate -= state.meta[old, Val(:fitness)] + d
                state.meta[old, Val(:npop)] -= 1
                Ntotal -= 1
            end
    
            genotype = state.meta[old, Val(:genotype)]
            if rand() < p_mu
                new_genotype = state.meta[end, Val(:genotype)] + 1
                if new_genotype != genotype
                    push!(state, new_genotype)
                    if makesnps
                        new_snps = copy(state.meta[old, Val(:snps)])
                        add_snps!(new_snps, mu, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
                        state.meta[end, Val(:snps)] = new_snps
                    end
                    state.meta[end, Val(:fitness)] = fitness(state, genotype, new_genotype)
                    push!(rates, 0.0)
                    add_edge!(state.phylogeny, nv(state.phylogeny), old)
                    old = length(state.meta)
                end
            end
            rates[old] += state.meta[old, Val(:fitness)]
            total_rate += state.meta[old, Val(:fitness)] + d
            state.meta[old, Val(:npop)] += 1
            Ntotal += 1

            # If carrying capacity is reached, one needs to die.
            if K < Ntotal
                while (die=sample(wnpops))!=old end
                rates[die] -= state.meta[die, Val(:fitness)]
                total_rate -= state.meta[die, Val(:fitness)] + d
                state.meta[die, Val(:npop)] -= 1
                Ntotal -= 1
            end
        end

        state.treal += -1.0 / total_rate * log(1.0 - rand())
        state.t += 1
    end
    if prune_on_exit
        prune_me!(state, mu)
    end
end

"""
    die_or_proliferate!(state::RealLattice{Int}; <keyword arguments>)

Moran-like dynamics on an spatially structured population. Each step is either a death or
(potential) birth and mutation event.

Individuals die at a rate `d`.
Birthrate depends linearily on the number of neighbors.

# Arguments
- `T::Int`: the number of steps to advance.
- `fitness`: function that assigns a fitness value to a genotype. Takes arguments `(state, old genotype, new_genotype)`.
- `p_grow=1.0`: Probability with which to actually proliferate. If no proliferation happens, mutation might still occur.
- `mu=0.0`: mutation rate.
- `mu_type=[:poisson, :fixed]`: Number of mutations is fixed, or Poisson-distributed.
- `genome_length=10^9`: Length of the haploid genome.
- `d=0.0`: death rate. Zero halts the dynamics after carrying capacity is reached.
- `baserate=1.0`: progressing real time is measured in `1/baserate`.
- `prune_period=0`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit=true`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function die_or_proliferate!(
    state::TumorConfiguration{G, <:RealLattice};
    fitness = (s, gold, gnew) -> 1.0,
    T = 0,
    mu::Float64 = 0.0,
    mu_type = :poisson,
    makesnps=true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    d::Float64 = 0.0,
    baserate = 1.0,
    p_grow = 1.0,
    constraint = true,
    prune_period = 0,
    prune_on_exit = true,
    callback = (s, t) -> begin end,
    abort = s -> false,
    sizehint=0,
    kwargs...) where G

    # genotypes = state.meta.genotypes
    # npops = state.meta.npops
    # fitnesses = state.meta.fitnesses
    # snps = state.meta.snps

    if sizehint > length(state.meta.genotypes)
        _resize!(state.meta, sizehint)
    end

    lattice = state.lattice
    I = CartesianIndices(lattice.data)
    Lin = LinearIndices(lattice.data)

    dim = dimension(lattice)
    sz = size(lattice)
    lin_N = sz[1]
    tot_N = length(lattice)

    fitness_lattice = [k != 0 ? getfitness(state, k) : 0.0 for k in lattice.data]
    br_lattice = zeros(Float64, size(lattice.data))

    nonzeros = count(x -> x != 0, lattice.data)
    base_br = 1.0 # - d

    nn = Lattices.Neighbors(lattice) # Initialize neighbors to the appr. type
    neighbor_indices = collect(1:length(nn))

    for k in 1:tot_N
        br_lattice[k]::Float64 = (1.0 - density!(nn, lattice, I[k])) * (base_br * fitness_lattice[k])
    end
    br_summary = reshape(sum(br_lattice, dims=1:dimension(lattice)-1), :)

    size_cross = prod(size(lattice)[1:end-1]) # number of elements in one slice

    p_mu = 1.0 - exp(-mu)

    new = 0
    new_cart = nn[1]
    old = 0
    selected = 0
    cumrate::Float64 = 0.0
    validneighbor = false
    action::Action = none
    total_rate::Float64 = mapreduce(+, enumerate(lattice.data)) do x
         x[2] > 0 ? d + br_lattice[x[1]] : 0.0
    end
    @debug total_rate

    @debug "Prune period is $prune_period"
    @inbounds for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            # @debug "Pruning..."
            prune_me!(state, mu)
        end
        # Base.invokelatest(callback, state, state.t)
        # if Base.invokelatest(abort, state)
        callback(state, state.t)
        if abort(state)
            break
        end

        # In case we pruned, renew bindings.
        # genotypes = state.meta.genotypes
        # npops = state.meta.npops
        # fitnesses = state.meta.fitnesses
        # snps = state.meta.snps
        ## Much cheaper than checking the whole lattice each iteration
        ## Leave the loop if lattice is empty
        if nonzeros == 0
            @info "Lattice empty. Exiting."
            break
        end
        if total_rate == 0.0
            continue
        end
        if total_rate < 0.0
            @warn("Total propensity became negative ($total_rate) after $(state.t) steps.")
            total_rate = 0.0
            continue
        end

        ########################
        ## REACTION SELECTION ##      
        ########################
        
        ## Die, proliferate or be dormant
        who_and_what = rand() * total_rate

        cumrate = 0.0
        selected = 1
        action = none        
        
        ## DEATH ##
        if who_and_what < d*nonzeros # death
            action = die
            _idx = floor(Int, who_and_what/d) # non-zero to die.
            while state[selected]==0 && selected != _idx && selected < tot_N
                selected += 1
            end
            z::Int = (selected-1)÷size_cross + 1 # `slice' coordinate
            ## DIE ##
            nonzeros -= 1
            total_rate -= br_lattice[selected] + d
            br_summary[z] -= br_lattice[selected]

            state[selected] = 0
            fitness_lattice[selected] = 0.
            br_lattice[selected] = 0.
            ## Update birth-rates
            neighbors!(nn, lattice, I[selected])
            for n in nn
                if !out_of_bounds(n, sz) && state[n] != 0
                    j = Lin[n]
                    local z::Int = (j-1)÷size_cross + 1
                    total_rate -= br_lattice[j]
                    br_summary[z] -= br_lattice[j]
                    br_lattice[j] +=  1.0 / nneighbors(lattice, n) * fitness_lattice[j] * base_br
                    total_rate += br_lattice[j]
                    br_summary[z] += br_lattice[j]
                end
            end
            ## END DIE ##
        ## PROLIFERATE ##
        else # birth/mutation
            who_and_what -= d*nonzeros
            action = proliferate
            slice = 1
            while cumrate + br_summary[slice] < who_and_what
                cumrate += br_summary[slice]
                selected += size_cross
                slice += 1
            end
            # @debug state.meta.misc["preselected"] = selected
            while cumrate+br_lattice[selected] < who_and_what && selected < tot_N
                cumrate += br_lattice[selected]
                selected += 1
            end
            if selected > tot_N
                error("selected too large: $selected")
            end
            z = selected÷size_cross + 1 # `slice' coordinate
            ## BIRTH & MUTATE ##
            old = selected
            new = old
            if !constraint
                new = 0
                while new != old
                    new = rand1toN(tot_N)
                end
            else
                b_grow = rand() < p_grow # Actual growth, or mutation only?
                if b_grow
                    neighbors!(nn, lattice, I[old])
                    validneighbor = false
                    for j in shuffle!(neighbor_indices)
                        nnj = nn[j]
                        if !out_of_bounds(nnj, sz) && state[nnj] == 0
                            new_cart = nnj
                            validneighbor = true
                            break
                        end
                    end
                    if !validneighbor
                        continue
                    end
                    nonzeros += 1
                    new = Lin[new_cart]
                end
                znew::Int = (new-1)÷size_cross + 1 # `slice' coordinate

                ## Mutation ##
                genotype = state[old]
                # @debug if genotype==0
                #     state.meta.misc["brlattice"] = br_lattice
                #     state.meta.misc["brslices"] = br_summary
                #     state.meta.misc["old"] = old
                #     state.meta.misc["totalrate"] = total_rate
                #     state.meta.misc["who_and_what"] = who_and_what
                #     state.meta.misc["z"] = z
                #     state.meta.misc["slice"] = slice
                #     state.meta.misc["cumrate"] = cumrate
                # end
                g_id::Int = gindex(state.meta, genotype)
                if !b_grow
                    state.meta[g_id, Val(:npop)] -= 1
                    # setnpop!(state, getnpop(state, g_id)-1, g_id)
                end
                if rand() < p_mu
                    ## TODO: custom genotypes
                    new_genotype = state.meta[end, Val(:genotype)] + 1 

                    if new_genotype != genotype
                        push!(state, new_genotype)
                        if makesnps
                            new_snps = copy(state.meta[g_id, Val(:snps)])
                            add_snps!(new_snps, mu, L = genome_length, kind=mu_type, allow_multiple = allow_multiple, replace = replace_mutations)
                            state.meta[end, Val(:snps)] = new_snps
                        end
                        state.meta[end, Val(:fitness)] = fitness(state, genotype, new_genotype)
                        add_edge!(state.phylogeny, nv(state.phylogeny), g_id)
                        genotype = new_genotype
                        g_id = lastindex(state.meta)
                    end
                end
                ## END Mutation ##
                @inbounds state[new] = genotype
                fitness_lattice[new] = state.meta[g_id, Val(:fitness)]

                if !b_grow
                    total_rate -= d + br_lattice[new]
                end
                br_summary[znew] -= br_lattice[new]
                br_lattice[new] = (1.0 - density!(nn, lattice, I[new])) * base_br * fitness_lattice[new]
                br_summary[znew] += br_lattice[new]
                total_rate += d + br_lattice[new]

                if b_grow
                    for n in nn
                        if !out_of_bounds(n, sz) && state[n] != 0
                            j = Lin[n]
                            local z::Int = (j-1)÷size_cross + 1
                            total_rate -= br_lattice[j]
                            br_summary[z] -= br_lattice[j]
                            br_lattice[j] -=  (1.0 / nneighbors(lattice, n)) * (fitness_lattice[j] * base_br)
                            br_summary[z] += br_lattice[j]
                            total_rate += br_lattice[j]
                        end
                    end
                end
            end
            ## END BIRTH & MUTATE ##
        end
        if action === none
            @debug "No reaction occured."
        end
        ##################
        ## END REACTION ##
        ##################


        state.t += 1
        state.treal += -1.0 / (baserate*total_rate) * log(1.0 - rand())
    end

    ## Update the phylogeny
    if prune_on_exit
        prune_me!(state, mu)
    end
    @debug "Done at $(state.t)"
end


"""
    twonew!(state::NoLattice{Int}; <keyword arguments>)

Moran-kind dynamics on an unstructured population. Each division results in two new genotypes.

Grow until carrying capacity
is reach. After that individuals begin replacing each other.

# Arguments
- `T::Int`: the number of steps to advance.
- `fitness`: function that assigns a fitness value to a genotype `g::Int`.
- `p_grow=1.0`: Probability with which to actually proliferate. If no proliferation happens, mutation might still occur.
- `mu`: mutation rate.
- `mu_type=[:poisson, :fixed]`: Number of mutations is fixed, or Poisson-distributed.
- `genome_length=10^9`: Length of the haploid genome.
- `d`: death rate.
- `baserate`: progressing real time is measured in `1/baserate`.
- `prune_period`: prune the phylogeny periodically after no. of steps.
- `prune_on_exit`: prune before leaving the simulation loop.
- `callback`: function of `state` and `time` to be called at each iteration.
    Used primarily for collecting observables during the run.
- `abort`: condition on `state` and `time` under which to end the run.
"""
function twonew!(
    state::TumorConfiguration{Int64, NoLattice{Int64}};
    fitness = (s, gold, gnew) -> 1.0,
    T = 0,
    mu::Float64 = 0.0,
    mu_type = :poisson,
    makesnps = true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    d::Float64 = 0.0,
    p_grow = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    DEBUG = false,
    callback = (s, t) -> begin end,
    abort = s -> false,
    K::Int64 = typemax(Int64), # carrying capacity
    sizehint=0, # initially resize state.meta to `sizehint`. do nothing if 0.
    kwargs...)

    rates = state.meta[:, :fitnesses] .* state.meta[:, :npops]

    ## sizehint ##
    if sizehint > length(state.meta.genotypes)
        _resize!(state.meta, sizehint)
    end

    Ntotal = total_population_size(state)
    total_rate = sum(rates) + d*Ntotal
    p_mu = 1.0 - exp(-mu)
    old = new1 = new2 = 0

    die!(;g) = die!(gindex(state.meta, g))
    function die!(gid)
        rates[gid] -= state.meta[gid, :fitness]
        total_rate -= state.meta[gid, :fitness] + d
        state.meta[gid, :npop] -= 1
        Ntotal -= 1
        nothing
    end
    function birth!(gid, g)
        new_genotype = state.meta[end, :genotype] + 1
        push!(state, new_genotype)
        state.meta[end, :fitness] = fitness(state, g, new_genotype)
        push!(rates, 0.0)
        add_edge!(state.phylogeny, nv(state.phylogeny), gid)
        new = length(state.meta)

        rates[new] += state.meta[new, :fitness]
        total_rate += state.meta[new, :fitness] + d
        state.meta[new, :npop] += 1
        Ntotal += 1
        return (new, new_genotype)
    end


    @inbounds for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            @debug "Pruning..."
            prune_me!(state, mu)
        end

        callback(state, state.t)
        if abort(state)
            break
        end

        wrates = Weights(rates, total_rate - d*Ntotal)
        wnpops = Weights((@view state.meta.npops[begin:state.meta._len]), Ntotal)

        
        ## Pick one to proliferate
        old = sample(wrates)
        genotype = state.meta[old, :genotype]
        
        b_die = rand() < d # are both daughter cell going to survive?
        
        die!(old)
        new1, _ = birth!(old, genotype)
        if !b_die
            new2, _ = birth!(old, genotype)
        end

        if makesnps
            new_snps = copy(state.meta[old, :snps])
            add_snps!(new_snps, mu, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
            state.meta[new1, :snps] = new_snps
            if !b_die
                new_snps = copy(state.meta[old, :snps])
                add_snps!(new_snps, mu, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
                state.meta[new2, :snps] = new_snps
            end
        end
        
        # If carrying capacity is reached, one needs to die.
        if K < Ntotal
            while (die=sample(wnpops))!=old end
            die!(die)
        end

        state.treal += -1.0 / total_rate * log(1.0 - rand())
        state.t += 1
    end
    if prune_on_exit
        prune_me!(state, mu)
    end
end



dynamics_dict = Dict(
    :moran => moran!,
    :die_or_proliferate => die_or_proliferate!,
)

## -- END module LatticeTumorDynamics --
end
