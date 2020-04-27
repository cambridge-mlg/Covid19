using Distributions

"""
    NegativeBinomial2(μ, ϕ)

Mean-variance parameterization of `NegativeBinomial`.

## Derivation
`NegativeBinomial` from `Distributions.jl` is parameterized following [1]. With the parameterization in [2], we can solve
for `r` (`n` in [1]) and `p` by matching the mean and the variance given in `μ` and `ϕ`.

We have the following two equations

(1) μ = r (1 - p) / p
(2) μ + μ^2 / ϕ = r (1 - p) / p^2

Substituting (1) into the RHS of (2):
  μ + (μ^2 / ϕ) = μ / p
⟹ 1 + (μ / ϕ) = 1 / p
⟹ p = 1 / (1 + μ / ϕ)
⟹ p = (1 / (1 + μ / ϕ)

Then in (1) we have
  μ = r (1 - (1 / 1 + μ / ϕ)) * (1 + μ / ϕ)
⟹ μ = r ((1 + μ / ϕ) - 1)
⟹ r = ϕ

Hence, the resulting map is `(μ, ϕ) ↦ NegativeBinomial(ϕ, 1 / (1 + μ / ϕ))`.

## References
[1] https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html
[2] https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

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
