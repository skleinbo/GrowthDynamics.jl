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
