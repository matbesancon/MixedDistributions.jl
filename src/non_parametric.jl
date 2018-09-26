import KernelDensity

"""
Detects mass points from a vector of data.

It uses a slope threshold on the ECDF:
the increase in ECDF in a range maxsupport*(maximum(xs) - minimum(xs))
has to be greater than threshold.

Returns a Pair of the vector of MassPoint and indices of elements of xs retained in mass points
"""
function find_peaks(xs::V; threshold = 0.1, maxsupport = 0.02) where {V<:AbstractVector{<:Real}}
    mass_points = Float64[]
    mass_probs  = Float64[]
    minval, maxval = extrema(xs)
    xcdf = ecdf(xs)
    Δ = maxsupport * (maxval - minval)
    mass_points = MassPoint[]
    mass_idxs   = BitSet([])
    for idx in eachindex(xs)
        if idx ∉ mass_idxs
            close_idxs = [jdx for jdx in eachindex(xs) if abs(xs[jdx]-xs[idx]) <= Δ]
            relweight = length(close_idxs) / length(xs)
            if relweight >= threshold
                union!(mass_idxs, close_idxs)
                push!(mass_points,MassPoint(mean(xs[jdx] for jdx in close_idxs), relweight))
            end
        end
    end
    return (mass_points, mass_idxs)
end

"""
Stores the information on a mass point: probability p and position x
"""
struct MassPoint
    x::Float64
    p::Float64
end