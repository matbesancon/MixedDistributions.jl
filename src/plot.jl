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
    return RecipesBase.plot(xaxis, cdf.(xaxis))
end
