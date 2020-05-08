using Turing, Distributions

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

"""
    generated_quantities(m::Turing.Model, c::MCMCChains.Chains)

Executes `m` for each of the samples in `c` and returns an array of the values returned by the `m` for each sample.

## Examples
Often you might have additional quantities computed inside the model that you want to inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()

    return interesting_quantity(θ, x)
end

m = demo(data)
chain = sample(m, alg, n)

# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples from the posterior/`chain`:
generated_quantities(m, chain) # <= results in a `Vector` of returned values from `interesting_quantity(θ, x)`
```
"""
function generated_quantities(m::Turing.Model, c::MCMCChains.Chains)
    # if `c` is multiple chains we pool them into a single chain
    chain = length(chains(c)) == 1 ? c : MCMCChains.pool_chain(c)

    varinfo = Turing.DynamicPPL.VarInfo(m)

    return map(1:length(chain)) do i
        Turing.DynamicPPL._setval!(varinfo, chain[i])
        m(varinfo)
    end
end

"Converts a vector of tuples to a tuple of vectors."
function vectup2tupvec(ts::AbstractVector{<:Tuple})
    k = length(first(ts))
    
    return tuple([[t[i] for t in ts] for i = 1:k]...)
end

"Converts a vector of named tuples to a tuple of vectors."
function vectup2tupvec(ts::AbstractVector{<:NamedTuple})
    ks = keys(first(ts))

    return (; (k => [t[k] for t in ts] for k ∈ ks)...)
end


"""
    rename!(d::Dict, names::Pair...)

Renames the keys given by `names` of `d`.
"""
function rename!(d::Dict, names::Pair...)
    # check that keys are not yet present before updating `d`
    for k_new in values.(names)
        @assert k_new ∉ keys(d) "$(k_new) already in dictionary"
    end

    for (k_old, k_new) in names
        d[k_new] = pop!(d, k_old)
    end
    return d
end

function arrarrarr2arr(a::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where {T<:Real}
    n1, n2, n3 = length(a), length(first(a)), length(first(first(a)))

    A = zeros(T, (n1, n2, n3))
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                A[i, j, k] = a[i][j][k]
            end
        end
    end

    return A
end

"""
    Chains(d::Dict)

Converts a `Dict` into a `Chains`, assuming the values of `d` is either
- `AbstractVector`: samples for a `Real` variable
- `AbstractMatrix`: samples for a `Vector` variable
- `AbstractArray{<:Any, 3}`: samples for a `Matrix` variable
"""
function MCMCChains.Chains(d::Dict)
    vals = []
    names = []

    for (k, v) in pairs(d)
        if v isa AbstractVector
            push!(vals, v)
            push!(names, k)
        elseif v isa AbstractMatrix
            push!(vals, v)
            append!(names, ["$(k)[$(i)]" for i = 1:size(v, 2)]) # assuming second dimension is dimensionality
        elseif v isa AbstractArray{<:Any, 3}
            indices = CartesianIndices(v[1, :, :])

            # The ordering is such that when you call `reshape` on the vector, the symbols will correspond with
            # the actual indices in the matrix, e.g. `X[i, j]` will be the same as `reshape(X, size)[i, j]`.
            for i = 1:size(indices, 2)
                for j = 1:size(indices, 1)
                    push!(vals, v[:, j, i])
                    push!(names, "$(k)[$j, $i]")
                end
            end
        else
            throw(ArgumentError("I'm so, so sorry but I can't handle $(typeof(v)) :("))
        end
    end

    return Chains(reduce(hcat, vals[2:end]; init = vals[1]), reduce(vcat, names[2:end]; init = names[1]))
end
