using Test
using GrowthDynamics

@testset "Lattices" begin
    include("latticestests.jl")
end
@testset "Geometry" begin
    include("geometrytests.jl")
end