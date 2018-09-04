using Test
using MixedDistributions
import Distributions
const Dst = Distributions
using Statistics: mean, var

const u1 = Dst.Uniform()
const u2 = Dst.Uniform(2.0,2.5)
const m = MixedDistribution([0.1,0.05,0.05],[1.,7.,3.],[0.2,0.6], [u1, u2])
const m1 = MixedDistribution(Float64[],Float64[],[0.4,0.6], [u1, u2])

@testset "Base interface" begin
    @test maximum(m1) ≈ maximum(u2)
    @test minimum(m1) ≈ minimum(u1)
    @test mean(m1)    ≈ 0.4 * mean(u1) + 0.6 * mean(u2)
    @test maximum(m)  ≈ 7.
    @test minimum(m)  ≈ 0.
end

@testset "Mixed properties" begin
    @test issorted(m.mass_points)
    @test mean(m) ≈ 0.2 * mean(u1) + 0.6 * mean(u2) + sum([0.1,0.05,0.05].*[1.,7.,3.])
end

@testset "Random generation" begin
	# each range is suppose to yield the corresponding mass point
	r1 = [0.0, 0.05,0.099, 0.1]
	r2 = [0.1025, 0.105,0.11,0.14]
	r3 = [0.151,0.16,0.19,0.2]
	# each range yields corresponding points on the uniform domains
	r4 = 0.2001:0.01:0.4
	r5 = 0.4001:0.01:1.0
	@test all(MixedDistributions.ordered_rand(m,r) ≈ 1.0 for r in r1)
	@test all(MixedDistributions.ordered_rand(m,r) ≈ 3.0 for r in r2)
	@test all(MixedDistributions.ordered_rand(m,r) ≈ 7.0 for r in r3)
	@test all(MixedDistributions.ordered_rand(m,r) ≈ ((r-0.2)/0.2) for r in r4)
	@test all(MixedDistributions.ordered_rand(m,r) ≈ ((r-0.4)/0.6)*0.5+2.0 for r in r5)
end

@testset "Homogeneous continuous behave like MixtureModel" begin
	mixture = Dst.MixtureModel([u1,u2],[0.4,0.6])
	@test mean(mixture) ≈ mean(m1)
	@test var(mixture) ≈  var(m1)
end
