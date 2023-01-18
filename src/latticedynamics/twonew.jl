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
