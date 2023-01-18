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
