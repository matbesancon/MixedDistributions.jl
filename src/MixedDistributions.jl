module MixedDistributions

import Base: maximum, minimum, rand
import Statistics
using Statistics: mean, var
using Random: AbstractRNG

using StatsBase: ecdf

import Distributions
const Dst = Distributions

export MixedDistribution

include("mixed_distribution.jl")
include("non_parametric.jl")

end # module
