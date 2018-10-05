module MixedDistributions

import Base: maximum, minimum, rand
import Statistics
using Statistics: mean, var, std
using Random: AbstractRNG

import Distributions
const Dst = Distributions

import RecipesBase

export MixedDistribution

include("mixed_distribution.jl")
include("plot.jl")

end # module
