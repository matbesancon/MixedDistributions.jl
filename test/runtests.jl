using Test
using MixedDistributions: MixedDistribution
import Distributions
const Dst = Distributions

const u1 = Dst.Uniform()
const u2 = Dst.Uniform(2.0,2.5)

@testset "Base interface" begin
    m = MixedDistribution(Float64[],Float64[],[0.4,0.6], [u1, u2])
    @test maximum(m) ≈ maximum(u2)
    @test minimum(m) ≈ minimum(u1)
end

@testset "Mixed properties" begin
    m = MixedDistribution([0.1,0.05,0.05],[1.,7.,3.],[0.2,0.6], [u1, u2])
    @test issorted(m.mass_points)
end