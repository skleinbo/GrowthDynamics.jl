module GrowthDynamics

export  Lattices,
        LatticeTumorDynamics

using Reexport

linspace(start,stop,length) = range(start, stop=stop, length=length)
DEBUG = false

include("Lattices.jl")
include("TumorConfigurations.jl")
@reexport using .TumorConfigurations
include("Phylogenies.jl")
include("LatticeTumorDynamics.jl")
include("TumorObservables.jl")
include("AnalysisMethods.jl")
include("FitnessIterators.jl")
include("PrettyPrinting.jl")


@reexport using LightGraphs
using FileIO
import Printf: @sprintf
using DataFrames

@reexport using ObservableCollector

using .FitnessIterators
@reexport using .LatticeTumorDynamics
@reexport using .TumorObservables
@reexport using .AnalysisMethods
@reexport using .Phylogenies


end
