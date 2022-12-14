module LatticeDynamics

import Distributions: Binomial, Exponential, cdf
import Graphs: nv, vertices, add_vertex!, add_edge!
using ..Lattices
import Random: shuffle!
using StatsBase: Weights, sample, mean
import ..Phylogenies: sample_ztp
import ..Populations: Population, add_genotype!, add_snps!, annotate_snps!, getfitness
import ..Populations: connect!, index, hassnps, lastgenotype, prune_phylogeny!, _resize!, rename!, snpsfrom
import ..Observables: total_population_size
import WeightedSampling: adjust_weight!, WeightedSampler, sample as smp, weight

export eden_with_density!, exponential!, moran!, twonew!

@enum Action none=0 proliferate=1 mutate=2 die=3

const MutationProfile = Tuple{Symbol,Float64,Int} # (:poisson/:fixed, rate, genome length)

function mu_func(mu)
    if mu isa Real
        function(args...)
            (mu, 1.0 - exp(-mu))
        end
    else
        function(args...)
            _mu = mu(args...)
            (_mu, 1.0-exp(-_mu))
        end
    end
end

###--- Start of simulation methods ---###

"""
    exponential!(state::NoLattice{Int}; <keyword arguments>)

Run exponential growth on an unstructered population until carrying capacity is reached.
No death. 

Each generation consists of the following actions for every genotype:

1. The number of decendents is drawn from a binomial distribution with parameters `n`, the 
   population number of that genotype, and `p` given by `1-exp(-f/<f>)` with fitness value `f`.
   That number as capped so as not to exceed the defined carrying capacity.
2. Of those decendents the number of mutants is drawn from a binomial distribution according to
   the mutation rate `mu`.

# Arguments
- `T::Int`: number of steps to advance.
- `fitness=(population, old_genotype, new_genotype)->1.0`: function that assigns a fitness value (default `1.0`) to a new genotype. 
- `mu=0.0`: mutation rate; either a function or number.
- `label=(population, old_genotype) -> lastgenotype(population)+1`: function that assigns the new genotype upon mutation. 
- `makesnps=true`: generate SNPs during the simulation?
- `mu_type=[:poisson, :fixed]`: number of mutations is fixed, or Poisson distributed.
- `genome_length=10^9`: length of the genome.
- `baserate=1.0`: progressing real time is measured in `1/baserate`.
- `prune_period=0`: prune the phylogeny periodically after no. of steps, set to `0` to disable.
- `prune_on_exit=true`: prune before leaving the simulation.
- `onstep`: callback that is executed at the very beginning of every time step.
- `ondeath`: callback that runs at the very end of a death event.
"""
exponential!(args...; mu=0.0, kwargs...) = _exponential!(args...; mu=mu_func(mu), kwargs...)
function _exponential!(
    state::Population{Int, NoLattice{Int}};
    T,
    K = 0, # Carrying capacity
    label = (s, gold) -> lastgenotype(s)+1,
    fitness = g -> 1.0,
    mu::Function,
    mu_type = :poisson,
    makesnps = true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    baserate = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    onstep = s->false,
    kwargs...)

    # P = state.phylogeny
    meta = state.meta

    @views npops = meta[:, Val(:npop)]
    @views fitness_vec = meta[:, Val(:fitness)]
    rates = fitness_vec .* npops
    wrates = Weights(rates)
    wnpops = Weights(npops)
    
    mean_fitness = mean(fitness_vec)
    Ntotal = sum(npops)
    total_rate = sum(rates)

    wrates = Weights(rates, total_rate)
    wnpops = Weights(npops, Ntotal)

    for step in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            @debug "Pruning..."
            prune_phylogeny!(state)
        end

        onstep(state) && break

        # Recalculate rates
        @views npops = meta[:, Val(:npop)]
        @views fitness_vec = meta[:, Val(:fitness)]
        rates = fitness_vec .* npops
        mean_fitness = mean(fitness_vec)
        Ntotal = sum(npops)
        total_rate = sum(rates)

        wrates = Weights(rates, total_rate)
        wnpops = Weights(npops, Ntotal)


        # need to collect, or new genotypes will be iterated too!
        for (g_old, j) in collect(pairs(meta.index))
            npop = meta[j, Val(:npop)]
            if npop == 0
                continue
            end
            max_nplus = K>0 ? K-Ntotal : typemax(Ntotal)-Ntotal
            if haskey(kwargs, :det_growth) && kwargs[:det_growth] == true
                pgrow = 1.0 # grow with certainty
                nplus = min(max_nplus, ceil(Int, npop * baserate))
            else
                pgrow = 1.0 - exp(-fitness_vec[j] / mean_fitness) # CDF of exponential with s/<s>
                nplus = min(max_nplus, rand(Binomial(npop, pgrow))) # How many proliferate?
            end
            # @debug "Pgrow = $pgrow"
            if nplus <= 0
                continue
            end
            
            thismu, p_mu = mu(state, g_old)
            nmutants = rand(Binomial(nplus, p_mu)) # How many of those mutate?
            for _ in 1:nmutants
                g_new = label(state, g_old)
                if g_new != g_old && fitness(state, g_old, g_new) != -Inf # -Inf indicates no mutation possible
                    idxnew = add_genotype!(state, g_new, g_old)

                    meta[idxnew, Val(:fitness)] = new_fitness = fitness(state, g_old, g_new)
                    meta[idxnew, Val(:npop)] = 1
                    if makesnps
                        new_snps = snpsfrom(state.meta, g_old)
                        add_snps!(new_snps, thismu; count = mu_type, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
                        state.meta[idxnew, Val(:snps)] = new_snps
                    end
                else # if mutation is impossible
                    nmutants -= 1
                end
            end
            meta[j, Val(:npop)] += nplus - nmutants
            Ntotal += nplus # keep track in the loop, because otherwise may overshoot K
        end

        state.treal += 1.0 / (baserate * mean_fitness)
        state.t += 1
    end
    if prune_on_exit
        prune_phylogeny!(state)
    end

    return nothing
end

"""
    moran!(state::NoLattice{Int}; <keyword arguments>)

(Generalized) Moran dynamics on an unstructured population. Birth and death events are independent
until the carrying capacity (keyword argument `K`) is reached. 
After that individuals begin replacing each other like in the classic Moran model.

# Arguments
- `T::Int`: the number of steps to advance.
- `d=0.0`: death rate.
- `K=0`: Carrying capacity. Set to `0` for unlimited.
- `fitness=(population, old_genotype, new_genotype)->1.0`: function that assigns a fitness value (default `1.0`) to a new genotype. 
- `mu=0.0`: mutation rate; either a function or number.
- `label=(population, old_genotype) -> lastgenotype(population)+1`: function that assigns the new genotype upon mutation. 
- `makesnps=true`: generate SNPs during the simulation?
- `mu_type=[:poisson, :fixed]`: number of mutations is fixed, or Poisson distributed.
- `genome_length=10^9`: length of the genome.
- `baserate=1.0`: progressing real time is measured in `1/baserate`.
- `prune_period=0`: prune the phylogeny periodically after no. of steps, set to `0` to disable.
- `prune_on_exit=true`: prune before leaving the simulation.
- `onstep`: callback that is executed at the very beginning of every time step.
- `onprebirth`: callback that runs if a birth event is about to occur. Return value determines whether cell actually proliferates.
- `onpostbirth`: callback that runs at the very end of a proliferation event.
- `ondeath`: callback that runs at the very end of a death event.
"""
moran!(args...; mu=0.0, kwargs...) = _moran!(args...; mu=mu_func(mu), kwargs...)
function _moran!(
    state::Population{Int, NoLattice{Int}};
    T,
    K::Int = 0, # carrying capacity
    label = (s, gold) -> lastgenotype(s)+1,
    fitness = (s, gold, gnew) -> 1.0,
    mu::Function,
    mu_type = :poisson,
    makesnps = true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    d = 0.0,
    baserate = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    onprebirth = (s, g_old)->true,
    onpostbirth = (s, g_old, g_new)->nothing,
    ondeath = (s, g)->nothing,
    onstep = s->false,
    kwargs...)

    rates = state.meta[:, Val(:fitness)] .* state.meta[:, Val(:npop)]

    Ntotal = total_population_size(state)
    total_rate = sum(rates) + d*Ntotal

    old = 0

    reaction::Action = none

    @inbounds for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            @debug "Pruning..."
            prune_phylogeny!(state)
        end

        onstep(state) && break

        wrates = Weights(rates, total_rate - d*Ntotal)
        wnpops = Weights((@view state.meta[:, Val(:npop)]), Ntotal)

        reaction = rand() < d / (total_rate/Ntotal) ? die : proliferate

        if reaction == die
            gid = sample(wnpops)
            g = state.meta[gid, :genotype]
            rates[gid] -= state.meta[g=g, Val(:fitness)]
            total_rate -= state.meta[g=g, Val(:fitness)] + d
            state.meta[g=g, Val(:npop)] -= 1
            Ntotal -= 1
            ondeath(state, g)
        else
            ## Pick one to proliferate
            old = sample(wrates)
            g_old = state.meta[old, Val(:genotype)]
            b_grow = onprebirth(state, g_old)
            if !b_grow
                rates[old] -= state.meta[g=g_old, Val(:fitness)]
                total_rate -= state.meta[g=g_old, Val(:fitness)] + d
                state.meta[g=g_old, Val(:npop)] -= 1
                Ntotal -= 1
            end
    

            thismu, p_mu = mu(state, g_old)
            if rand() < p_mu
                g_new = label(state, g_old)
                if g_new != g_old
                    add_genotype!(state, g_new, g_old)
                    if makesnps
                        new_snps = snpsfrom(state.meta, g_old)
                        add_snps!(new_snps, thismu; count = mu_type, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
                        state.meta[end, Val(:snps)] = new_snps
                    end
                    state.meta[end, Val(:fitness)] = fitness(state, g_old, g_new)
                    push!(rates, 0.0)
                    old = lastindex(state.meta)
                end
            else
                g_new = g_old
            end
            rates[old] += state.meta[g=g_old, Val(:fitness)]
            total_rate += state.meta[g=g_old, Val(:fitness)] + d
            state.meta[g=g_old, Val(:npop)] += 1
            Ntotal += 1

            # If carrying capacity is reached, one needs to die.
            if Ntotal>K>0
                to_die = sample(wnpops)
                rates[to_die] -= state.meta[to_die, Val(:fitness)]
                total_rate -= state.meta[to_die, Val(:fitness)] + d
                state.meta[to_die, Val(:npop)] -= 1
                Ntotal -= 1
            end
            onpostbirth(state, g_old, g_new)
        end

        state.treal += -log(1.0 - rand())/(baserate*total_rate)
        state.t += 1
    end
    if prune_on_exit
        prune_phylogeny!(state)
    end
end

"""
    eden_with_density!(state::Population)

Cells proliferate to neighboring lattice sites with a rate proportional to the number of 
free neighboring sites. A time step is either a birth, death, or mutation-only event.

A custom `label` function must be provided if genotypes are not integers.

See the 'extended help' for signatures of the various callbacks.

# Keyword arguments
- `T::Int`: number of steps to advance.
- `d=0.0`: death rate. Zero halts the dynamics after carrying capacity is reached.
- `fitness=(population, old_genotype, new_genotype)->1.0`: function that assigns a fitness value (default `1.0`) to a new genotype. 
- `mu=0.0`: mutation rate; either a function or number.
- `label=(population, old_genotype) -> lastgenotype(population)+1`: function that assigns the new genotype upon mutation. 
- `makesnps=true`: generate SNPs during the simulation?
- `mu_type=[:poisson, :fixed]`: number of mutations is fixed, or Poisson distributed.
- `genome_length=10^9`: length of the genome.
- `baserate=1.0`: progressing real time is measured in `1/baserate`.
- `prune_period=0`: prune the phylogeny periodically after no. of steps, set to `0` to disable.
- `prune_on_exit=true`: prune before leaving the simulation.
- `onstep`: callback that is executed at the very beginning of every time step.
- `onprebirth`: callback that runs if a birth event is about to occur. Return value determines whether cell actually proliferates.
- `onpostbirth`: callback that runs at the very end of a proliferation event.
- `ondeath`: callback that runs at the very end of a death event.

# Extended help

## Mutations

The rate of mutations is steered by the keyword argument `mu`, which is either a number or a function.

A number is automatically converted to an appropriate function.
As a function it must have signature `mu(population, old_genotype, I_old, I_new)->(rate, p_mutate)::Tuple{Float64, Float64}` where
* `old_genotype` is the genotype of the parent cell
* `I_old`, `I_new` are the *linear* lattice indices of the parent and daughter cell.
This way the mutation rate can depend on position as well as on mutations present in the
parental genome.

The output must be a tuple, where the first entry `rate` determines the number of mutations **if** mutations happen;
the probability of which is given by the second entry `p_mutate`. The main reason for keeping these quantities separate is that
one will often not generate SNPs during the simulation, because that process is rather costly, but generate them later on the
basis of the final phylogeny.

If `mu` is a number, it is implicitely wrapped in a function that returns `(mu, 1-exp(-mu))`, i.e. the mutation probability is
the probability of at least one event under a Poisson distribution with rate `mu`.

Setting `makesnps` to `true/false` determines whether SNPs are generated during the simulation. If it is set to `false`, the first value
returned by `mu` is inconsequential.

Finally, `label(population, old_genotype)->new_genotype` assigns the designation to a newly entering genotype. It defaults to numbering genotypes
consecutively. See also [`rename!`](@ref).

## Fitness

When a new genotype enters due to a mutation event, it is assigned a fitness value
given by a user-provides function `(population, old_genotype, new_genotype)->Float64` passed as
keyword argument `fitness`. For example, to inherit the fitness, one would provide 
`fitness=(p,og,ng)->p.meta[g=og; :fitness]`.

## Callbacks

Callbacks are triggered at certain stages of the simulation loop:

* `onstep(population)::Bool` is executed at the beginning of each time step. If `true` is returned, the simulation ends.
* `onprebirth(population, Iold)::Bool` is executed when a birth event is triggered, but *before* it is executed. If `false` is returned, the cell
   will not proliferate, but might still mutate.
* `onpostbirth(population, Iold, Inew)` is executed after proliferation and mutation are completed. Useful for collecting observables.
* `ondeath(population, Idead)` is executed when a death event has finished.
"""
eden_with_density!(args...; mu=0.0, kwargs...) = _eden_with_density!(args...; mu=mu_func(mu), kwargs...)
function _eden_with_density!(
    state::Population{G, <:RealLattice};
    T,
    label = (s, gold) -> lastgenotype(s)+1,
    fitness = (s, gold, gnew) -> 1.0,
    mu::Function,
    mu_type = :poisson,
    makesnps = true,
    genome_length = 10^9,
    replace_mutations = false,
    allow_multiple = false,
    d = 0.0,
    baserate = 1.0,
    prune_period = 0,
    prune_on_exit = true,
    onprebirth = (s, Iold)->true,
    onpostbirth = (s, Iold, Inew)->nothing,
    ondeath = (s, I)->nothing,
    onstep = s->false,
    sizehint = 0,
    strict = false,
    kwargs...) where G

    if sizehint > length(state.meta.genotype)
        _resize!(state.meta, sizehint)
    end

    lattice = state.lattice
    I = CartesianIndices(lattice.data)
    Lin = LinearIndices(lattice.data)

    sz = size(lattice)
    tot_N = length(lattice)

    # TODO: Should require only one pass.
    free_neighbors = [ count(x->!out_of_bounds(lattice, x) && state[x]==zero(G), neighbors(lattice, y)) for y in I ]
    fitness_lattice = [k != zero(G) ? getfitness(state, k) : 0.0 for k in lattice.data]
    dr_lattice = [k != zero(G) ? d : 0.0 for k in lattice.data]
    br_lattice = zeros(Float64, size(lattice.data))

    nonzeros = count(x -> x != zero(G), lattice.data)
    base_br = 1.0

    nn = Lattices.Neighbors(lattice) # Initialize neighbors to the appr. type
    neighbor_indices = collect(1:length(nn))

    for k in 1:tot_N
        br_lattice[k] = free_neighbors[k]/nneighbors(lattice, I[k]) * (base_br * fitness_lattice[k])
    end
    br_sampler = WeightedSampler(reshape(br_lattice, :))
    dr_sampler = WeightedSampler(reshape(dr_lattice, :))

    new = 0
    new_cart = nn[1]
    old = 0
    selected = 0
    validneighbor = false
    action::Action = none
    total_rate::Float64 = br_sampler.heap[1] + dr_sampler.heap[1]
    @debug total_rate

    @debug "Prune period is $prune_period"
    for t in 0:T
        if prune_period > 0 && state.t > 0 && (state.t) % prune_period == 0
            prune_phylogeny!(state)
        end

        onstep(state) && break

        total_rate = br_sampler.heap[1] + dr_sampler.heap[1]
        ## Much cheaper than checking the whole lattice each iteration
        ## Leave the loop if lattice is empty
        if nonzeros == 0
            @info "Lattice empty. Exiting."
            break
        end
        if total_rate == 0.0
            @info "Total rate is zero. Lattice full and no death possible? Exiting." maxlog=1
            break
        elseif total_rate < 0.0
            error("""Total propensity became negative ($total_rate) after $(state.t) steps.
                     This should not happen.""")
        end

        ########################
        ## REACTION SELECTION ##      
        ########################
        
        ## Die, proliferate or be dormant
        who_and_what = rand() * total_rate

        selected = 1
        action = none        
        
        ## DEATH ##
        if who_and_what < dr_sampler.heap[1] # death
            action = die
            selected = smp(dr_sampler, 1)
            ## DIE ##
            nonzeros -= 1

            state[selected] = zero(G)
            fitness_lattice[selected] = 0.0
            adjust_weight!(dr_sampler, selected, 0)
            adjust_weight!(br_sampler, selected, 0)
            ## Update birth-rates
            neighbors!(nn, lattice, I[selected])
            for n in nn
                if !out_of_bounds(n, sz)
                    j = Lin[n]
                    free_neighbors[j] += 1
                    if  state[n] != zero(G)
                        adjust_weight!(br_sampler, j,  free_neighbors[j] / nneighbors(lattice, n) * fitness_lattice[j] * base_br)
                    end
                end
            end
            ondeath(state, selected)
            ## END DIE ##
        ## PROLIFERATE ##
        else # birth/mutation
            nonzeros==0 && continue
            action = proliferate
            selected = smp(br_sampler, 1)
            # @debug state.meta.misc["preselected"] = selected
            if selected > tot_N
                error("selected too large: $selected")
            end
            ## BIRTH & MUTATE ##
            old = selected
            new = old
            b_grow = onprebirth(state, old) # Actual growth, or mutation only?
            if b_grow
                neighbors!(nn, lattice, I[old])
                validneighbor = false
                for j in shuffle!(neighbor_indices)
                    nnj = nn[j]
                    if !out_of_bounds(nnj, sz) && state[nnj] == zero(G)
                        new_cart = nnj
                        validneighbor = true
                        break
                    end
                end
                if !validneighbor
                    @warn "No valid neighbor found." I[old] nn
                        strict && begin 
                        global dbg =Dict(:br => br_sampler, :fn => free_neighbors)
                        
                        throw(ErrorException("No valid neighbor found."))
                    end
                    continue
                end

                nonzeros += 1
                new = Lin[new_cart]
            end

            ## MUTATION ##
            genotype = state[old]
            g_id::Int = index(state.meta, genotype)
            if !b_grow
                state.meta[g_id, Val(:npop)] -= 1
            end
            thismu, p_mu = mu(state, genotype, old, new)
            if rand() < p_mu
                new_genotype = label(state, g_id)

                if new_genotype != genotype
                    add_genotype!(state, new_genotype, genotype)
                    if makesnps
                        new_snps = snpsfrom(state.meta, genotype)
                        add_snps!(new_snps, thismu, L = genome_length, count=mu_type, allow_multiple = allow_multiple, replace = replace_mutations)
                        state.meta[end, Val(:snps)] = new_snps
                    end
                    state.meta[end, Val(:fitness)] = fitness(state, genotype, new_genotype)
                    genotype = new_genotype
                    g_id = lastindex(state.meta)
                end
            end
            ## END Mutation ##
            @inbounds state[new] = genotype
            fitness_lattice[new] = state.meta[g_id, Val(:fitness)]

            adjust_weight!(br_sampler, new,  free_neighbors[new]/nneighbors(lattice, I[new]) * base_br * fitness_lattice[new])
            adjust_weight!(dr_sampler, new,  d)

            if b_grow
                neighbors!(nn, lattice, I[new])
                for n in nn
                    if !out_of_bounds(n, sz)
                        j = Lin[n]
                        free_neighbors[j] -= 1
                        if state[n] != zero(G)
                            adjust_weight!(br_sampler, j,  free_neighbors[j]/nneighbors(lattice, n) * fitness_lattice[j] * base_br)
                        end
                    end
                end
            end
            ## END BIRTH & MUTATE ##
            onpostbirth(state, old, new)
        end
        if action === none
            @debug "No reaction occured."
        end
        ##################
        ## END REACTION ##
        ##################


        state.t += 1
        state.treal += -log(1.0 - rand())/(baserate*total_rate)
    end

    ## Update the phylogeny
    if prune_on_exit
        prune_phylogeny!(state)
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
    state::Population{Int, NoLattice{Int}};
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
    K::Int = typemax(Int), # carrying capacity
    kwargs...)

    rates = state.meta[:, :fitness] .* state.meta[:, :npop]

    Ntotal = total_population_size(state)
    total_rate = sum(rates) + d*Ntotal
    p_mu = 1.0 - exp(-mu)
    old = new1 = new2 = 0

    die!(;g) = die!(index(state.meta, g))
    function die!(gid)
        rates[gid] -= state.meta[gid, Val(:fitness)]
        total_rate -= state.meta[gid, Val(:fitness)] + d
        state.meta[gid, Val(:npop)] -= 1
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
            prune_phylogeny!(state)
        end

        callback(state, state.t)
        if abort(state)
            break
        end

        wrates = Weights(rates, total_rate - d*Ntotal)
        wnpops = Weights((@view state.meta[:, Val(:npop)]), Ntotal)

        
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
            new_snps = isnothing(state.meta[old, :snps]) ? Int[] : copy(state.meta[old, :snps])
            add_snps!(new_snps, mu, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
            state.meta[new1, :snps] = new_snps
            if !b_die
                new_snps = isnothing(state.meta[old, :snps]) ? Int[] : copy(state.meta[old, :snps])
                add_snps!(new_snps, mu, L = genome_length, allow_multiple = allow_multiple, replace = replace_mutations)
                state.meta[new2, :snps] = new_snps
            end
        end
        
        # If carrying capacity is reached, one needs to die.
        if K < Ntotal
            die = sample(wnpops)
            die!(die)
        end

        state.treal += -1.0 / total_rate * log(1.0 - rand())
        state.t += 1
    end
    if prune_on_exit
        prune_phylogeny!(state)
    end
end

const dynamics_dict = Dict(
    :moran => moran!,
    :eden_with_density => eden_with_density!,
)

## -- END module LatticeDynamics --
end