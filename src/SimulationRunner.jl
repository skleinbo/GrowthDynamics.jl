module SimulationRunner
#=
Helper functions to save results and parameters
of a run.
=#
using MacroTools
using Distributed
using JSON
using DataFrames
import GrowthDynamics.AnalysisMethods: timeseries

export  get_last_file_number,
        json_parameters,
        _run_sim_conditional!,
        repeated_runs,
		JobRunner,
		@parameters,
		setup_simulation_environment,
		stop, start, pause, stop!, empty!, run!


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

include("ParameterTemplate.jl")

mutable struct Job
	id::Int
	job::Base.Generator
	status::Symbol ## Either of :RUNNABLE, :RUNNING, :DONE, :FAILED, :PAUSED
end
Job(job::Base.Generator) = Job(0, job, :RUNNABLE)

mutable struct Worker
	pid::Int
	commands::RemoteChannel
	consumer::Future
end
Worker(pid::Int) = Worker(pid, RemoteChannel(()->Channel{Any}(Inf)), Future())

mutable struct JobRunner
	workers::Vector{Worker} ## Which workers does the runner utilize?

	jobs::Vector{Job}

	running::Bool

	jobs_channel::RemoteChannel ## Channel to fill with parameters for every run
	results_channel::RemoteChannel	 ## Channel to be filled with results

	feeder_task::Task
end

function descend_dicts(l, r)
	if isa(l, Dict) && isa(r, Dict)
		merge(descend_dicts, l, r)
	else
		return r
	end
end


macro parameters(ex)
	ex = MacroTools.prewalk(rmlines, ex)
	# @show ex
	# @show ex.head ex.args length(ex.args)
	if length(ex.args) != 2 || ex.head != :for
		error("Usage: @parameters for [...] end")
	end
	parameters = map(x->(x.args[1], x.args[2]), ex.args[1].args)
	template = ex.args[2].args[1]
	base_template = length(ex.args[2].args) >= 2 ? ex.args[2].args[2] : SimulationRunner.parameter_template

	gen_expr = Expr(:generator, :(merge(SimulationRunner.descend_dicts, $base_template, $template)), (ex.args[1].args)...)

	quote
		Job($(esc(gen_expr)))
	end
end



module tmp
	import ..SimulationRunner
	p = (SimulationRunner.@parameters for N = [1,2,3], mu in [1e-2]
		Dict(
			:dyn => Dict(
				:T => 10*N
			),
			:simulation => Dict(
				:N => N
			)
		)
		parameter_template
	end)
end


function setup_simulation_environment(num_cpu::Integer=Sys.CPU_THREADS)
	workers = addprocs(num_cpu)
	setup_simulation_environment(workers)
end

function setup_simulation_environment(workers::Vector)
	## Code loading on the workers/master
	@show workers
	@eval Main begin
		using Distributed
		using DataFrames
		using BSON
		import Base.Iterators: repeated, flatten, product
		@everywhere $workers begin
			# import Serialization: serialize
			import ProgressMeter
			import Printf: @sprintf
			using GrowthDynamics
			using GrowthDynamics.TumorConfigurations
		end
	end
	jr = JobRunner(map(pid->Worker(pid), workers),
		Job[],
		false,
		RemoteChannel(()->Channel{Any}(Inf)),
		RemoteChannel(()->Channel{Any}(Inf)),
		Task(nothing))
	jr.feeder_task = feeder(jr)
	for worker in 1:length(workers)
		start_job_processing(jr, worker)
	end

	return jr
end

function process_jobs(jobs::RemoteChannel, results::RemoteChannel, commands::RemoteChannel)
	run = false
	job = nothing
	try
	while true
		if isready(commands)
			cmd = take!(commands)
			if cmd == :RUN
				@info "Running."
				run = true
			elseif cmd == :ABORT || !isopen(results) || !isopen(jobs)
				@info "Aborting."
				break
			elseif cmd == :PAUSE
				@info "Pausing."
				run = false
			end
		end
		if run && isopen(jobs)
			job = take!(jobs)
			@info "Received a job on $(myid())."
				SimulationRunner._run_sim_conditional!(
					job[:simulation],
					job[:dyn],
					results
				    )
		else
			@info "Waiting for command..."
			wait(commands)
		end
	end
	catch e
		@error e
		@info job
		rethrow(e)
	end
	return :SUCCESS
end

function start_job_processing(runner::JobRunner, id::Int)
	pid = runner.workers[id].pid
	runner.workers[id].consumer = @spawnat pid begin
		@async SimulationRunner.process_jobs(runner.jobs_channel, runner.results_channel, runner.workers[id].commands)
	end
	@info "Consumer task launched on $pid."
end

function submit!(runner::JobRunner, job::Job)
	try
		job.id = maximum(map(x->x.id, runner.jobs)) + 1
	catch
		job.id = 1
	end

	push!(runner.jobs, job)
	nothing
end

function feeder(runner::JobRunner)
	@task begin
		while runner.running
			job_id = findfirst(x->x.status == :QUEUING, runner.jobs)
			if job_id != nothing
				for job in runner.jobs[job_id].job
					if !isopen(runner.jobs_channel)
						break
					end
					put!(runner.jobs_channel, job)
					@info "Fed $job_id into queue."
					yield()
				end
				@info "Setting $job_id to :RUNNING"
				runner.jobs[job_id].status = :RUNNING
			end
			sleep(0.5)
		end
	end
end

function run!(runner::JobRunner)
	for j in runner.jobs
		if j.status == :RUNNABLE
			@info "Setting job to :QUEUING"
			j.status = :QUEUING
		end
	end
	if istaskdone(runner.feeder_task)
		runner.feeder_task = feeder(runner)
	end
	for w in 1:length(runner.workers)
		pid = runner.workers[w].pid
		# if (remotecall_fetch(()->!isdefined(Main, :process_task) || istaskdone(process_task), runner.workers[w].pid))
		fut = runner.workers[w].consumer
		if @fetchfrom pid istaskdone(fetch(fut))
			@info "Starting consumer task on $pid"
			start_job_processing(runner, w)
		end
		put!(runner.workers[w].commands, :RUN)
	end
	runner.running = true
	try
		if !istaskstarted(runner.feeder_task)
			schedule(runner.feeder_task)
		end
		@info "Feeder task started."
	catch err
		@info "Replacing job feeder."
		runner.feeder_task = feeder(runner)
	end
end

send_cmd(runner::JobRunner, id, cmd::Symbol) = begin
	if 0 < id <= length(runner.workers)
		put!(runner.workers[id].commands, cmd)
	end
end

stop(runner::JobRunner, id) = send_cmd(runner, id, :ABORT)
pause(runner::JobRunner, id) = send_cmd(runner, id, :PAUSE)
start(runner::JobRunner, id) = send_cmd(runner, id, :RUN)

function stop!(R::JobRunner)
	@sync for id in 1:length(R.workers)
		@async begin
			stop(R, id)
			wait(R.workers[id].consumer)
		end
	end

	R.running = false
	@info "Waiting for feeder task to end."
	wait(R.feeder_task)
end

function empty!(R::JobRunner)
	empty!(R.jobs)
	while isready(R.jobs_channel)
		take!(R.jobs_channel)
	end
end




function _run_sim_conditional!(
	sim_params,
	dyn_params,
    results_channel,
    )

	verbosity = sim_params[:verbosity]
	dyn! = sim_params[:dynamics]
	setup = sim_params[:setup]
	sweep = sim_params[:sweep]
	obs = sim_params[:obs]
	abort = sim_params[:abort]


	verbosity>1 && println(params)

    state = Base.invokelatest(setup)
    max_T = dyn_params[:T]
    if sweep>0
        dyn!(state;dyn_params...,T=sweep)
    end

    k = 2
    X = []

    obs_callback(s,t) = Base.invokelatest(obs, X,s,t)

    dyn!(state;dyn_params...,T=max_T,callback=obs_callback,abort=abort)

    if isempty(X)
        if isdefined(Main,:STRICT) && Main.STRICT
            global state = state
            error("No observables collected!")
        else
            @warn ("No observables collected!")
        end
    end
    put!(results_channel, (:N =>sim_params[:N], :s => dyn_params[:s], :mu => dyn_params[:mu],
     :d => dyn_params[:d], :f => dyn_params[:f], X))
end

"To be deprecated"
function repeated_runs(N::Integer, dyn!, obs, setup, parameters;
     results_channel,
     sweep=0,
     abort=(s,t)->false,
     verbosity=0,
     callback=()->begin end
    )
    verbosity > 1 && println("repeated_runs\t $params")

    # initial_params = deepcopy(params)
    for n in 1:N
        _run_sim_conditional!(dyn!, obs, setup, parameters; results_channel=results_channel,
            sweep=sweep,abort=abort)
        callback()
    end
end

function collect_results!(df::Ref{DataFrame}, rc::Distributed.RemoteChannel, maxtakes=0)
	n = 0
	while n < maxtakes
		raw = take!(rc)
		if raw == :end
			return df[]
		end
		try
			## The last entry of the returned tuple contains by convention
			## the observables
			parameters = raw[1:end-1]
			obs = raw[end] |> timeseries
			D = merge(Dict(parameters), obs)
			if isempty(df[])
				df[] = DataFrame(typeof.(values(D)), collect(keys(D)), 0)
			end
			push!(df[], merge(Dict(parameters), obs))
		catch err
			@error err
			return df[]
		end
		n += 1
	end
	df[]
end


##==END Module==##
end
