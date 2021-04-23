import LinearAlgebra: det

@testset "Plane" begin
    @test_throws ArgumentError Plane((1,1,1), (2,2))
    P = Plane((0,0,0), (1,1,1), (1,2,1))
    @test isvalid(P)
    @test det([P.u P.v P.w]) ≈ +1.0
    P = Plane((0,0,0), -[1,1,1])
    @test det([P.u P.v P.w]) ≈ +1.0
    p = @SVector [0,0,0]
    u = 1/sqrt(3)* @SVector [1,1,1]
    v = 1/sqrt(6)* @SVector [1,2,1]
    w = 1/sqrt(2)* @SVector [-1,0,1]
    @test Plane((0,0,0), (1,1,1), (1,2,1)) == 
        Plane(p, u,v,w)
end