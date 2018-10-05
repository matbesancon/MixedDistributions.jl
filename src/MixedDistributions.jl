module MixedDistributions

import Base: maximum, minimum, rand
import Statistics
using Statistics: mean, var
using Random: AbstractRNG

import Distributions
const Dst = Distributions

export MixedDistribution

include("mixed_distribution.jl")

end # module
