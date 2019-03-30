module SimulationRunner
#=
Helper functions to save results and parameters
of a run.
=#
using JSON
export  get_last_file_number,
        json_parameters,
        _run_sim_conditional!,
        repeated_runs


function get_last_file_number(path::AbstractString)
    old_num::Int = 0
    for file in readdir(path)
        new_num = 0
        f_match = match(r"^(\d+)_\d+\.bin*$", file)
        if f_match != nothing
            new_num = parse(Int,f_match.captures[1])
            if new_num > old_num
                old_num = new_num
            end
        end
    end
    return old_num + 1
end

function json_parameters(p)
    p = deepcopy(p)
    ## Cleanup
    # if typeof(p[:dyn][:state]) <: OffLattice.FreeSpace
    #     p[:dyn][:state] = join([p[:dyn][:state].MaxPopulation, typeof(p[:dyn][:state])],",")
    # elseif typeof(p[:dyn])
    #     p[:dyn][:state] = join([p[:dyn][:state].Na, typeof(p[:dyn][:state])],",")
    # end
    delete!(p[:dyn],:f_mut)
    delete!(p[:simulation],:obs)
    delete!(p[:simulation],:abort)
    try
        json(p,4)
    catch err
        @show p
        throw(err)
    end
end
## END helper


function _run_sim_conditional!(
    dyn!,
    obs,
    setup,
    params;
    sweep=0,
    abort=(s,t)->false,
    verbosity=0
    )

    verbosity>1 && println(params)

    state = setup()
    max_T = params[:T]
    dyn!(state;params...,T=sweep)

    k = 2
    X = []

    obs_callback(s,t) = obs(X,s,t)

    dyn!(state;params...,T=max_T,callback=obs_callback,abort=abort)

    if isempty(X)
        if isdefined(Main,:STRICT) && Main.STRICT
            global state = state
            error("No observables collected!")
        else
            @warn ("No observables collected!")
        end
    end
    X
end


function repeated_runs(N::Integer, dyn!, obs, setup, parameters;
     sweep=0,
     abort=(s,t)->false,
     verbosity=0,
     callback=()->begin end
    )
    verbosity > 1 && println("repeated_runs\t $params")

    # initial_params = deepcopy(params)
    X = []
    for n in 1:N
        push!(X, _run_sim_conditional!(dyn!, obs, setup, parameters;
            sweep=sweep,abort=abort
        ))
        callback()
    end

    return X
end

##==END Module==##
end
