module GrowthDynamics

export  Lattices,
        LatticeTumorDynamics

using Reexport
using Distributed

# if !isdefined(Main, :NWORKER)
#     global const NWORKER = 4
# end

# if nworkers()<NWORKER
#     addprocs(NWORKER-nworkers()+ (workers()[1]==1 ? 1 : 0))
# end

linspace(start,stop,length) = range(start, stop=stop, length=length)
DEBUG = false

# @everywhere begin
# if isdefined(Main, :DEBUG)
#      DEBUG=$DEBUG
# else
#      DEBUG=false
# end
# end


include("Lattices.jl")
include("TumorConfigurations.jl")
@reexport using .TumorConfigurations
include("Phylogenies.jl")
include("LatticeTumorDynamics.jl")
include("TumorObservables.jl")
include("AnalysisMethods.jl")
include("FitnessIterators.jl")
include("PrettyPrinting.jl")

include("SimulationRunner.jl")
@reexport using .SimulationRunner

using Serialization
@reexport using LightGraphs
using FileIO
using JSON: json
import Printf: @sprintf
using DataFrames

@reexport using ObservableCollector

using .FitnessIterators
@reexport using .LatticeTumorDynamics
@reexport using .TumorObservables
@reexport using .AnalysisMethods
@reexport using .Phylogenies


end
