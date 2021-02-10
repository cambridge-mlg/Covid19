using Distributions, CUDA


struct NegativeBinomialVectorized{T1, T2} <: DiscreteMultivariateDistribution
    r::T1
    p::T2
end


function NegativeBinomialVectorized2(μ, ϕ)
    p = @. 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomialVectorized(r, p)
end


function Distributions.logpdf(d::NegativeBinomialVectorized, k::AbstractArray{<:Real, 1})
    Mod_p = (d.p isa CuArray) ? CUDA : Base
    Mod_kr = (d.r isa CuArray || k isa CuArray) ? CUDA : Base

    r = d.r .* Mod_p.log.(d.p) .+ k .* Mod_p.log1p.(-d.p)
    return sum(r .- Mod_kr.log.(k .+ d.r) .- SpecialFunctions.logbeta.(d.r, k .+ 1))
end
