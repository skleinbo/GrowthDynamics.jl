using Test
using GrowthDynamics

@testset begin
@testset "Lattices" begin
    include("latticestests.jl")
end
@testset "Geometry" begin
    include("geometrytests.jl")
end
@testset "TumorConfigurations" begin
    include("tumorconfigurations.jl")
end
@testset "LatticeTumorDynamics" begin
    include("latticetumordynamics.jl")
end
end