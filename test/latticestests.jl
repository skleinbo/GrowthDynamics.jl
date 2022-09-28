## TEST SUIT for the Lattices submodule ##

using GrowthDynamics.Lattices
using GeometryBasics
# LineLattice
@testset "LineLattice" begin
    lattice = Lattices.LineLattice(1/2, fill(0, 5))
    @test size(lattice) == (5,)
    @test length(lattice) == 5
    @test coord(lattice, (3,)) == 3*lattice.a
    @test coord(lattice, CartesianIndex(3)) == 3/2
    @test index(lattice, 0.3) == CartesianIndex(1)
    I = CartesianIndices(lattice.data)
    C = coord.(Ref(lattice), I)
    I′ = index.(Ref(lattice), C)
    @test all(I .== I′)
    @testset "Neighbors" begin
        @test neighbors(lattice, (3,)) == [CartesianIndex(2), CartesianIndex(4)]
        @test nneighbors(lattice, (3,)) == 2
        @test nneighbors(lattice, (1,)) == 1
        @test nneighbors(lattice, (5,)) == 1
    end
    @test dist(lattice, (5,), (6,)) == lattice.a
    @test (lattice[3] = 1) == 1
    @test density(lattice, (2,)) == 1/2
    I = filter(x->!out_of_bounds(x, size(lattice)), neighbors(lattice, (2,)))
    lattice[I] .= 1
    @test lattice[I] == [1,1]
    @test density(lattice, (2,)) == 1.0
end
# HexagonalLattice
@testset "HexagonalLattice" begin
    lattice = Lattices.HexagonalLattice(1/2, fill(0, 5,5))
    @test size(lattice) == (5,5)
    @test length(lattice) == 25
    @test coord(lattice, (1,1)) == Point2f(0.0, 0.0)
    @test (coord(lattice, (5,5)) .≈ Point2f(4*lattice.a, 4*sin(pi/3)*lattice.a)) |> all
    @test (coord(lattice, (4,4)) .≈ ((3+1/2)*lattice.a, 3*sin(pi/3)*lattice.a)) |> all
    @test (coord(lattice, CartesianIndex(4,4)) .≈ ((3+1/2)*lattice.a, 3*sin(pi/3)*lattice.a)) |> all
    @test Tuple(index(lattice, ((3+1/2)*lattice.a+0.01, 3*sin(pi/3)*lattice.a+0.02))) == (4,4)
    I = CartesianIndices(lattice.data)
    C = coord.(Ref(lattice), I)
    I′ = index.(Ref(lattice), C)
    @test all(I .== I′)
    @testset "Neighbors" begin
        @test nneighbors(lattice, (1,1)) == 2
        @test nneighbors(lattice, (5,5)) == 2
        @test nneighbors(lattice, (3,3)) == 6
    end
    @test dist(lattice, CartesianIndex(1,1), CartesianIndex(5,5)) ≈ lattice.a*sqrt(4^2 + 4^2*sin(pi/3)^2)
    @test (lattice[3,3] = 1) == 1
    @test density(lattice, (3,2)) == 1/6
    I = filter(x->!out_of_bounds(x, size(lattice)), neighbors(lattice, (2,2)))
    lattice[I] .= 1
    @test lattice[I] == fill(1, length(I))
    @test density(lattice, (2,2)) == 1.0
end
# CubicLattice
@testset "CubicLattice" begin
    lattice = Lattices.CubicLattice(1/2, fill(0, 5,5,5))
    @test size(lattice) == (5,5,5)
    @test length(lattice) == 125
    @test coord(lattice, (1,1,1)) == Point3f(0.0, 0.0, 0.0)
    @test (coord(lattice, (5,5,5)) .≈ lattice.a.*Point3f(4,4,4)) |> all
    @test (coord(lattice, CartesianIndex(4,4,4)) .≈ lattice.a.*(3,3,3)) |> all
    @test Tuple(index(lattice, (3,4.5,2.1))) == (7,10,5)
    I = CartesianIndices(lattice.data)
    C = coord.(Ref(lattice), I)
    I′ = index.(Ref(lattice), C)
    @test all(I .== I′)
    @testset "Neighbors" begin
        @test nneighbors(lattice, (1,1,1)) == 3
        @test nneighbors(lattice, (5,5,5)) == 3
        @test nneighbors(lattice, (3,3,3)) == 6
    end
    @test dist(lattice, CartesianIndex(1,1,0), CartesianIndex(5,5,0)) ≈ lattice.a*sqrt(2)*4
    @test (lattice[3,3,3] = 1) == 1
    @test density(lattice, (3,2,3)) == 1/6
    I = filter(x->!out_of_bounds(x, size(lattice)), neighbors(lattice, (2,2,2)))
    lattice[I] .= 1
    @test lattice[I] == fill(1, length(I))
    @test density(lattice, (2,2,2)) == 1.0
end
# FCCLattice
@testset "FCCLattice" begin
    lattice = Lattices.FCCLattice(1/2, fill(0, 10,10,10))
    @test size(lattice) == (10,10,10)
    @test length(lattice) == 10^3
    @test coord(lattice, (1,1,1)) == Point3f(0.0, 0.0, 0.0)
    @test (coord(lattice, (1,2,4)) .≈ lattice.a.*Point3f(0,1+1/2,1+1/2)) |> all
    @test (coord(lattice, (1,2,5)) .≈ lattice.a.*Point3f(0,1,2)) |> all
    @test (coord(lattice, (1,3,4)) .≈ lattice.a.*Point3f(0,2+1/2,1+1/2)) |> all
    @test (coord(lattice, (1,3,5)) .≈ lattice.a.*Point3f(0,2,2)) |> all
    I = CartesianIndices(lattice.data)
    C = coord.(Ref(lattice), I)
    I′ = index.(Ref(lattice), C)
    @test all(I .== I′)

    @test (coord(lattice, CartesianIndex(4,4,4)) .≈ lattice.a.*(1+1/2,3,1+1/2)) |> all
    @test Tuple(index(lattice, Point3f(3,4.5,2.1))) == (13, 10, 9)
    @testset "Neighbors" begin
        @test_broken  nneighbors(lattice, (1,1,1)) == 3
        @test nneighbors(lattice, (5,5,5)) == 12
        @test_broken nneighbors(lattice, (3,3,3)) == 6
    end
    @test dist(lattice, CartesianIndex(1,1,1), CartesianIndex(5,5,1)) ≈ lattice.a*sqrt(2^2+4^2)
    @test (lattice[3,3,3] = 1) == 1
    @test density(lattice, (3,3,2)) == 1/12
    I = filter(x->!out_of_bounds(x, size(lattice)), neighbors(lattice, (2,2,2)))
    lattice[I] .= 1
    @test lattice[I] == fill(1, length(I))
    @test density(lattice, (2,2,2)) == 1.0
end
