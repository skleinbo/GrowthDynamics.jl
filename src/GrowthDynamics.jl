module GrowthDynamics

export  Lattices,
        LatticeTumorDynamics

using Reexport
using Distributed
using OpenCLPicker

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

@opencl begin
DEVICE_ID = OpenCLPicker.devicePicker()
@everywhere begin
    DEVICE_ID=$DEVICE_ID
    import OpenCLPicker
    (cl_device, cl_ctx, cl_queue) = OpenCLPicker.provideContext(DEVICE_ID)
end
end

include("Lattices.jl")
include("TumorConfigurations.jl")
include("Phylogenies.jl")
include("LatticeTumorDynamics.jl")
include("TumorObservables.jl")
include("AnalysisMethods.jl")
include("FitnessIterators.jl")

@opencl begin
    include("OffLattice.jl")
    include("OffLatticeTumorDynamics.jl")
end
include("SimulationRunner.jl")
@reexport using .SimulationRunner

using Serialization
@reexport using LightGraphs
@reexport using MetaGraphs
using FileIO
using JSON: json
import Printf: @sprintf
using DataFrames

@opencl using .OffLatticeTumorDynamics
@reexport using ObservableCollector

using .FitnessIterators
@reexport using .TumorConfigurations
@reexport using .LatticeTumorDynamics
@reexport using .TumorObservables
@reexport using .AnalysisMethods
@reexport using .Phylogenies


end
