using Distributions

"""
    NegativeBinomial2(μ, σ²)

Mean-variance parameterization of `NegativeBinomial`.

## References
- https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
- https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + ϕ / μ)
    r = ϕ

    # # TODO remove
    # (p ≤ .0 || p > 1.) && @info "r < 0: $(ReverseDiff.value(r)) for inputs" ReverseDiff.value(μ) ReverseDiff.value(ϕ)

    # @info "NegativeBinomial2" ReverseDiff.value(σ²) ReverseDiff.value(μ) ReverseDiff.value(ϕ) ReverseDiff.value(p) ReverseDiff.value(r)

    return NegativeBinomial(r, p)
end


"""
    GammaMeanCv(mean, cv)

Mean-variance-coefficient parameterization of `Gamma`.

## References
- https://www.rdocumentation.org/packages/EnvStats/versions/2.3.1/topics/GammaAlt
"""
function GammaMeanCv(mean, cv)
    k = cv^(-2)
    θ = mean / k
    return Gamma(k, θ)
end
