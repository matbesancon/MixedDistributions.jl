module MixedDistributions

import Base: maximum, minimum
import Statistics: mean

import Distributions
const Dst = Distributions

export MixedDistribution, MassPoint

"""
    A mass point in a discrete or mixed distribution is represented by its probability p and position x
"""
struct MassPoint
    p::Float64
    x::Float64
end
# TODO replace p and x vecs with MassPoint

"""
    A mixed continuous-discrete distribution
"""
struct MixedDistribution{DT<:Dst.ContinuousUnivariateDistribution} <: Dst.ContinuousUnivariateDistribution
    mass_probs::Vector{Float64}
    mass_points::Vector{Float64}
    cont_weights::Vector{Float64}
    cont_dists::Vector{DT}
    function MixedDistribution(mass_probs::Vector{Float64}, mass_points::Vector{Float64}, cont_weights::Vector{Float64}, cont_dists::Vector{DT}) where {DT <: Dst.ContinuousUnivariateDistribution}
        length(mass_probs) == length(mass_points)  || ArgumentError("Mass length inconsistency")
        length(cont_weights) == length(cont_dists) || ArgumentError("Continuous length inconsistency")
        # sort mass points
        pairs = sort(zip(mass_points, mass_probs) |> collect, by = t -> t[1])
        mass_points = [t[1] for t in pairs]
        mass_probs =  [t[2] for t in pairs]
        sum(mass_probs) + sum(cont_weights) ≈ one(eltype(mass_probs)) && return new{DT}(mass_probs, mass_points, cont_weights, cont_dists)
        @warn "Prior probabilities not normalized"
        s = sum(mass_probs) + sum(cont_weights)
        new{DT}(mass_probs./s, mass_points, cont_weights./s, cont_dists)
    end
end

function maximum(md::MixedDistribution)
    max_mass = isempty(md.mass_points) ? -Inf64 : maximum(md.mass_points)
    max_cont = isempty(md.cont_dists)  ? -Inf64 : maximum.(md.cont_dists) |> maximum
    return max(max_mass, max_cont)
end

function minimum(md::MixedDistribution)
    min_mass = isempty(md.mass_points) ? +Inf64 : minimum(md.mass_points)
    min_cont = isempty(md.cont_dists)  ? +Inf64 : minimum.(md.cont_dists) |> minimum
    return min(min_mass, min_cont)
end

function Dst.ncategories(md::MixedDistribution)
    return length(md.mass_probs)
end

function mean(md::MixedDistribution)
    mass_part = sum(md.mass_points.*md.mass_probs)
    cont_part = sum(w * mean(d) for (w,d) in zip(md.cont_weights, md.cont_dists))
    return mass_part + cont_part
end

function Dst.insupport(md::MixedDistribution, x::Real)
    minimum(md) <= x && x <= maximum(md)
end

function Dst.pdf(md::MixedDistribution, x::Real)
    any(md.mass_points .≈ x) && DomainError("PDF undefined at a mass point") 
end

function Dst.cdf(md::MixedDistribution, x::Real)
    mass_part = sum(pᵢ for (pᵢ, xᵢ) in zip(md.mass_probs, md.mass_points) if xᵢ <= x)
    cont_part = sum(μᵢ * Dst.cdf(dᵢ, x) for (dᵢ, μᵢ) in (md.cont_dists, md.cont_weights))
    return mass_part + cont_part
end

Dst.ncomponents(md::MixedDistribution) = Dst.ncategories(md) + length(md.cont_dists)

Dst.probs(md::MixedDistribution) = vcat(md.mass_probs, md.cont_weights)

end # module
