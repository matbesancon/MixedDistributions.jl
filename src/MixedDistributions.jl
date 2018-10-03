module MixedDistributions

import Base: maximum, minimum, rand
import Statistics
using Statistics: mean, var, std
using Random: AbstractRNG

import Distributions
const Dst = Distributions

import RecipesBase

export MixedDistribution

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
        s = sum(mass_probs) + sum(cont_weights)
        s ≈ one(eltype(mass_probs)) && return new{DT}(mass_probs, mass_points, cont_weights, cont_dists)
        @warn "Prior probabilities not normalized"
        new{DT}(mass_probs./s, mass_points, cont_weights./s, cont_dists)
    end
end

function Base.maximum(md::MixedDistribution)
    max_mass = isempty(md.mass_points) ? -Inf64 : maximum(md.mass_points)
    max_cont = isempty(md.cont_dists)  ? -Inf64 : maximum.(md.cont_dists) |> maximum
    return max(max_mass, max_cont)
end

function Base.minimum(md::MixedDistribution)
    min_mass = isempty(md.mass_points) ? +Inf64 : minimum(md.mass_points)
    min_cont = isempty(md.cont_dists)  ? +Inf64 : minimum.(md.cont_dists) |> minimum
    return min(min_mass, min_cont)
end

function Dst.ncategories(md::MixedDistribution)
    return length(md.mass_probs)
end

function Statistics.mean(md::MixedDistribution)
	μ = sum(md.mass_points.*md.mass_probs)
	for idx in eachindex(md.cont_dists)
		μ += mean(md.cont_dists[idx]) * md.cont_weights[idx]
	end
	return μ
end

"""
	Variance of mixed continuous discrete computed as a mixture:
	Var[X] = E[Var[X|θ]] + Var[E[X|θ]]
"""
function Statistics.var(md::MixedDistribution)
	disc_weight = sum(md.mass_probs)
    μ_discrete = sum(md.mass_points.*md.mass_probs)/disc_weight
    σ2d = sum((md.mass_points.-μ_discrete).^2 .* md.mass_probs)
    μ = mean(md)
    tot_var = isempty(md.mass_probs) ? 0.0 : disc_weight * (σ2d + μ_discrete^2)
    for idx in eachindex(md.cont_dists)
		tot_var += md.cont_weights[idx] * (var(md.cont_dists[idx]) + mean(md.cont_dists[idx])^2)
    end
    return tot_var - μ^2
end

function Dst.insupport(md::MixedDistribution, x::Real)
    return minimum(md) <= x && x <= maximum(md)
end

function Dst.pdf(md::MixedDistribution, x::Real)
    any(md.mass_points .≈ x) && DomainError("PDF undefined at a mass point") 
end

function Dst.cdf(md::MixedDistribution, x::Real)
    cdf_point = 0.0
    for idx in eachindex(md.mass_probs)
        if md.mass_points[idx] <= x
            cdf_point += md.mass_probs[idx]
        end
    end
    for idx in eachindex(md.cont_dists)
        cdf_point += md.cont_weights[idx] * Dst.cdf(md.cont_dists[idx], x)
    end
    return cdf_point
end

"""
	Number of components of the mixed distribution:
	number of discrete points + continuous distributions
"""
Dst.ncomponents(md::MixedDistribution) = Dst.ncategories(md) + length(md.cont_dists)

"""
	Returns a vector of prior probabilities [discrete priors, continuous priors]
"""
Dst.probs(md::MixedDistribution) = vcat(md.mass_probs, md.cont_weights)

Base.rand(rng::AbstractRNG, md::MixedDistribution) = ordered_rand(md, rand(rng))

"""
	Ordered_rand returns the quantity corresponding to
	an already-generated random number
"""
function ordered_rand(md::MixedDistribution, r)
	cs_discrete = cumsum(md.mass_probs)
	for idx in eachindex(cs_discrete)
		if r <= cs_discrete[idx]
			return md.mass_points[idx]
		end
	end
	prob_offset = isempty(cs_discrete) ? 0.0 : last(cs_discrete)
	cs_cont = vcat(cs_discrete, cumsum(md.cont_weights) .+ prob_offset)
	for idx in length(cs_discrete)+1:length(cs_cont)
		if r <= cs_cont[idx]
			cont_idx = idx-length(cs_discrete)
			last_prob = idx == 1 ? 0.0 : cs_cont[idx-1]
			scaled_point = (r - last_prob) / (cs_cont[idx] - last_prob)
			return Dst.quantile(md.cont_dists[cont_idx], scaled_point)
		end
	end
	# all points reached beforehand
	throw(CDFException(md))
end

"""
	Exception used when Cumulated Distribution Function does not integrate to 1 in a computation  
"""
struct CDFException{D<:Dst.Distribution} <: Exception
	d::D
	CDFException(d::D) where {D<:Dst.Distribution} = new{D}(d) 
end

"""
Graphically represents a MixedDistribution structure, use the keyword func = :cdf
to represent the CDF instead of PDF
"""
function RecipesBase.plot(md::MixedDistribution; func = :pdf, fillalpha = 0.4, xlow = -Inf64, xhigh = Inf64)
    if func == :cdf
        return plot_cdf(md, xlow, xhigh)
    end
    p = RecipesBase.plot()
    for (wi,di) in zip(md.cont_weights, md.cont_dists)
        wf = x -> wi * Dst.pdf(di,x)
        m = mean(di)
        s = std(di)
        xaxis = max(m-5.0*s,minimum(di), xlow):0.01:min(m+5.0*s,maximum(di), xhigh) |> collect
        RecipesBase.plot!(p, xaxis, wf.(xaxis), fillalpha = fillalpha, fillcolor = :match, fillrange = 0)
    end

    for (xi,pi) in zip(md.mass_points,md.mass_probs)
        RecipesBase.plot!(p, [xi, xi], [0.0, pi], line = :arrow, color = :blue)
    end
    return p
end

function plot_cdf(md::MixedDistribution, xlow, xhigh; shallow = false)
    xlow_trunc  = max(xlow,  mean(md)-5.0*std(md))
    xhigh_trunc = min(xhigh, mean(md)+5.0*std(md))
    xaxis = collect(minimum(md.mass_points):0.01:maximum(md.mass_points))
    for itr in (max(minimum(md),xlow_trunc):0.01:first(md.mass_points),last(md.mass_points):0.01:min(maximum(md),xhigh_trunc))
        append!(xaxis, itr)
    end
    sort!(xaxis)
    cdf = x -> Dst.cdf(md, x)
    return Plots.plot(xaxis, cdf.(xaxis))
end

end # module
