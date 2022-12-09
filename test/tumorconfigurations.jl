import Dictionaries: IndexError
using .Lattices
import Graphs: Edge, edges, nv
import GrowthDynamics.TumorConfigurations: MetaDatum
using GrowthDynamics.TumorConfigurations

@testset "TumorConfigurations" begin
    @testset "MetaData" begin
        MD = MetaDatum((1,1024,1.0,[1,2,8],(1,1e-8)))
        @test MD.npop == 1024
        @test_throws MethodError MetaDatum((1,1024,1,[1,2,8],(1,1e-8))) # fitness is not a Float64
        @test eltype(MetaData(Int64).genotype) == Int64
        @test MetaData(MD).npop == [MD.npop]
        @test_throws ArgumentError MetaData([1,2],[10,20,30])
        @test sum(MetaData([1,2,3],[10,20,30]).npop) == 60

        M = MetaData(["3","2","1"],[10,20,30])
        @test M[1].genotype == "3"
        @test M[g="1"].npop == 30
        @test_throws MethodError M[g=1]
        @test_throws ArgumentError M[g="4"]

        @test_throws BoundsError M[4] = MetaDatum("4")
        @test_throws ErrorException M[1, :notafield] = "TEST"
        @test (M[3, :genotype] = "3+") == "3+" && M[3].genotype == "3+"
        @test_throws IndexError M[2, :genotype] = "3+"
        @test M[3, :genotype] == "3+"
        @test M[g="3+", :genotype] == "3+"
        @test M[:genotype, g="3+"] == "3+"
        push!(M, "P5")
        @test M[g="P5"].npop == 0
    end
    @testset "TumorConfigurations" begin
        NLconf = nolattice_state()[1]
        @test NLconf.meta.npop == [1]
        push!(NLconf, 2)
        @test NLconf.meta.npop == [1,0]
        for l in [LineLattice, HexagonalLattice, CubicLattice]
            conf = single_center(l, 9; g1=1, g2=2)[1]
            lat = conf.lattice
            dim = dimension(lat)
            @test sum(conf.lattice.data) == length(lat)+1
            conf[fill(1, dim)...] = 2
            @test sum(conf.lattice.data) == length(lat)+2
            @test_throws ArgumentError conf[fill(2, dim)...] = 3
            push!(conf, 3)
            conf[fill(2, dim)...] = 3
            @test sum(conf.lattice.data) == length(lat)+2+2
            @test nv(conf.phylogeny) == 3
            conf[fill(9, dim)...] = 0
            @test sum(conf.meta[:, :npop]) == length(lat)-1
            @test (conf[:] .= 1).meta[g=1, :npop] == length(conf.lattice)
            ##
            for f in [i//10 for i in 1:10]
                conf = half_space(l, 9; f=f, g1=1, g2=2)[1]
                @test conf.meta.npop == floor.(Int, length(conf.lattice).*[1-f, f])
            end
        end
        # remove_genotype!
        conf = uniform(CubicLattice, 31; g=0)[1]
        for g in 1:5
            push!(conf, g)
        end
        connect!(conf, 2 => 1)
        connect!(conf, 3 => 2)
        connect!(conf, 4 => 2)
        connect!(conf, 5 => 1)
        for g in 1:5
            conf[g, :, :] .= g
        end
        @test conf.meta[:, :npop] == fill(1024, 5)
        @test_throws ArgumentError remove_genotype!(conf, 1)
        @test remove_genotype!(conf, 2)
        @test_throws ArgumentError getnpop(conf, 2)
        @test Edge(3=>1) in edges(conf.phylogeny)

    end
end
