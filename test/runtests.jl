using Test
using MixedDistributions: MixedDistribution
import Distributions
const Dst = Distributions

const u1 = Dst.Uniform()
const u2 = Dst.Uniform(2.0,2.5)

@testset "Base interface" begin
    m1 = MixedDistribution([],[],[0.4,0.6], [u1, u2])
    @test maximum(m1) ≈ maximum(u2)
    @test minimum(m1) ≈ minimum(u1)     
end