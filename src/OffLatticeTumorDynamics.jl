module OffLatticeTumorDynamics

import GeometryTypes:   Point2f0,
                        Point3f0
import LightGraphs: vertices, edges, add_edge!, add_vertex!
import .TumorConfigurations

import OffLattice:  FreeSpace,
                    set_birthrates!,
                    new_position,
                    random_config,
                    totalrate,
                    new_cell!,
                    delete_cell!,
                    sync_to_host!,
                    sync_to_device

export die_or_proliferate!


function die_or_proliferate!(
    ;state::FreeSpace=random_config(0),
    fitness=Float64[],
    T=0,
    mu::Float64=0.0,
    f_mut=(L,G,g)->maximum(vertices(G))+1,
    d::Float32=1/100f0,
    constraint=true,
    weight_params=Dict(:rCell=>0f0,:sigma=>1f0, :max_density=>12f0),
    DEBUG=false,
    kwargs...)

    new = 0
    old = 0
    selected = 0
    cumrate = 0.
    action = :none
    total_rate = totalrate(state)
    nonzeros = countnz(state._mask)
    DEBUG && @show kw;

    # max_density = Float32(weight_params[:max_density] * exp(-weight_params[:rCell]/weight_params[:sigma]^2))

    @inbounds for t in 1:T
        ## Much cheaper than checking the whole lattice each iteration
        ## Leave the loop if lattice is empty
        if nonzeros==0
            break
        end

        ## Die, proliferate or be dormant
        who_and_what = rand()*total_rate

        cumrate = 0f0
        selected = 0
        action = :none
        while selected < state.MaxPopulation
            selected += 1
            if !Bool(state._mask[selected])
                continue
            end
            cumrate += state.deathrates[selected]
            if cumrate > who_and_what
                action = :die
                break
            end
            cumrate += state.birthrates[selected]
            if cumrate > who_and_what
                action = :proliferate
                break
            end
        end


        if action == :die
            DEBUG && println("Die")
            nonzeros -= 1
            delete_cell!(state, selected)
            ## Update birth-rates
        elseif action == :proliferate && nonzeros < state.MaxPopulation
            DEBUG && println("Live")
            old = selected

            old_genotype = state.genotypes[old]
            new_genotype = old_genotype
            if rand()<mu && new_genotype<=length(fitness)
                new_genotype = f_mut(state,state.Phylogeny,lattice.data[old])
                add_vertex!(G,new_genotype)
                add_edge!(G,(new_genotype,old_genotype))
                ## TODO: time of mutation => make T a field of FS
            end
            new_pos = new_position(Point2f0(state.positions[:,old]), 2*weight_params[:rCell], 0.)
            new_cell!(state, new_pos, new_genotype; d=state.deathrates[old])

            nonzeros += 1
        else
            DEBUG && println("Noone")
        end
        set_birthrates!(
            Val{false},   # false = use OpenCL
            state; weight_params...
        )
        sync_to_host!(state)
        total_rate = totalrate(state)
    end
end

dynamics_dict = Dict(
    :die_or_proliferate => OffLatticeTumorDynamics.die_or_proliferate!
)

end
