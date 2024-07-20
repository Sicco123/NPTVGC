using NPTVGC
using Test

@testset "NPTVGC.jl" begin
    # Write your tests here.

    @test NPTVGC.weights!((1, 1), 1.0, "e", "smoothing") == [1]
    
end
