module GrowthDynamics

export  Lattices,
        LatticeTumorDynamics

import Printf: @sprintf

using DataFrames
using FileIO
using Graphs

linspace(start,stop,length) = range(start, stop=stop, length=length)
DEBUG = false

include("Lattices.jl")
include("RoughInterfaces.jl")
include("Phylogenies.jl")
include("TumorConfigurations.jl")
# @reexport using .TumorConfigurations
include("TumorObservables.jl")
include("LatticeTumorDynamics.jl")
include("AnalysisMethods.jl")
include("PrettyPrinting.jl")

# @reexport using .LatticeTumorDynamics
# @reexport using .TumorObservables
# @reexport using .AnalysisMethods
# @reexport using .Phylogenies

end
