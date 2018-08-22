module SimpleDiscretes

import Base: minimum, maximum
import Statistics

import Distributions
const Dst = Distributions

export SimpleDiscrete

struct SimpleDiscrete{T1<:Real, T2<:Real} <: Dst.DiscreteUnivariateDistribution
    xs::Vector{T1}
    ps::Vector{T2}
end

maximum(d::SimpleDiscrete) = isempty(d.xs) ? -Inf64 : maximum(d.xs)
minimum(d::SimpleDiscrete) = isempty(d.xs) ? +Inf64 : minimum(d.xs)

Statistics.mean(d::SimpleDiscrete) = sum(md.mass_points .* md.mass_probs)
Statistics.var(d::SimpleDiscrete)  = Statistics.mean(d.xs.^2) - Statistics.mean(d.xs)^2

Dst.insupport(d::SimpleDiscrete, x::Real) = minimum(d) <= x && x <= maximum(d)

function Dst.cdf(d::SimpleDiscrete, x::Real)
    return sum(p_i for (x_i, p_i) in zip(md.mass_points, md.mass_probs) if x_i <= x)
end

function Dst.pdf(d::SimpleDiscrete{T1,T2}, x::Real)
    for (x\_i,p\_i) in zip(d.xs,d.ps)
        x\_i \approx x && return p_i    
    end
    return zero(T2)
end

Dst.ncomponents(d::SimpleDiscrete) = length(d.xs)

Dst.params(d::SimpleDiscrete) = (d.xs, d.ps)

end