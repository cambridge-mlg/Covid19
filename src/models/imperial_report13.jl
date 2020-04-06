# Replication of https://github.com/ImperialCollegeLondon/covid19model
using Turing
using Distributions, StatsBase
using ArgCheck

"""
    NegativeBinomial2(μ, σ²)

Mean-variance parameterization of `NegativeBinomial`.

## References
- https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
- https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    # ϕ = μ^2 / (σ² - μ)
    # p = 1 / ϕ
    # r = μ^2 / (σ² - μ) = μ^2 / (μ^2 / ϕ) = ϕ


    σ² = μ + (μ^2 / ϕ)
    p = (σ² - μ) / σ²
    r = μ^2 / (σ² - μ)

    # @info "NegativeBinomial2" ForwardDiff.value(σ²) ForwardDiff.value(μ) ForwardDiff.value(ϕ) ForwardDiff.value(p) ForwardDiff.value(r)

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

######################
## Model definition ##
######################
@model imperial_model_report13(
    num_countries,     # [Int] num. of countries
    num_impute,        # [Int] num. of days for which to impute infections
    num_obs_countries, # [Vector{Int}] days of observed data for country `m`; each entry must be ≤ N2
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    ::Type{TV} = Vector{Float64}
) where {TV} = begin
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)

    # Check args
    @argcheck size(num_obs_countries) == (num_countries, )
    @argcheck size(cases) == (num_countries, )
    @argcheck all([size(cases[i]) == (num_total_days, ) for i = 1:num_countries])
    @argcheck size(deaths) == (num_countries, )
    @argcheck all([size(deaths[i]) == (num_total_days, ) for i = 1:num_countries])
    @argcheck size(π) == (num_countries, )
    @argcheck all([size(π[i]) == (num_total_days, ) for i = 1:num_countries])
    @argcheck length(covariates) == num_countries
    @argcheck all([size(covariates[m]) == (num_total_days, num_covariates) for m = 1:num_covariates])
    @argcheck size(epidemic_start) == (num_countries, )
    @argcheck size(serial_intervals) == (num_total_days, )

    # Sample variables
    τ ~ Exponential(0.03)
    y ~ arraydist([Exponential(1. / τ) for m = 1:num_countries])
    ϕ ~ truncated(Normal(0, 5), 1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
    κ ~ truncated(Normal(0, 0.5), 1e-6, 100) # In Stan they don't make this truncated, but specify that `κ ≥ 0` and so it will be transformed
    # κ ~ Turing.Bijectors.transformed(Normal(0, 0.5), Turing.Bijectors.Exp{0}())
    μ ~ product_distribution(truncated.(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries)), 0, Inf))
    # μ ~ Turing.Bijectors.transformed(product_distribution(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries))), Turing.Bijectors.Exp{1}())
    α ~ product_distribution([GammaMeanCv(.5, 1) for i = 1:num_covariates])

    # Transforming variables
    daily_cases_pred = [TV(undef, num_total_days) for m = 1:num_countries]
    expected_daily_deaths = [TV(undef, num_total_days) for m = 1:num_countries]
    Rₜ = [TV(undef, num_total_days) for m = 1:num_countries]

    for m = 1:num_countries
        # country-specific parameters
        πₘ = π[m]
        daily_cases_pred_m = daily_cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rₜ_m = Rₜ[m]

        # Imputation of `num_impute` first days (since we're looking quite far into the past)

        ### Stan-equivalent ###
        # daily_cases_pred[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        # Rt[,m] = mu[m] * exp(covariate1[,m] * (-alpha[1]) + covariate2[,m] * (-alpha[2]) +
        #                      covariate3[,m] * (-alpha[3])+ covariate4[,m] * (-alpha[4]) + covariate5[,m] * (-alpha[5]) + 
        #                      covariate6[,m] * (-alpha[6])); // + GP[i]); // to_vector(x) * time_effect
        daily_cases_pred_m[1:num_impute] .= y[m]
        Rₜ_m .= μ[m] * exp.(- covariates[m] * α)

        ### Stan-equivalent ###
        # for (i in (N0+1):N2) {
        #     convolution=0;
        #     for(j in 1:(i-1)) {
        #         convolution += daily_cases_pred[j, m]*SI[i-j]; // Correctd 22nd March
        #     }
        #     daily_cases_pred[i, m] = Rt[i,m] * convolution;
        # }
        for t = (num_impute + 1):num_total_days
            daily_cases_pred_m[t] = Rₜ_m[t] * sum([daily_cases_pred_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1)])
        end

        ### Stan-equivalent ###
        # E_deaths[1, m]= 1e-9;
        # for (i in 2:N2){
        #     E_deaths[i,m]= 0;
        #     for(j in 1:(i-1)){
        #         E_deaths[i,m] += daily_cases_pred[j,m]*f[i-j,m];
        #     }
        # }
        expected_daily_deaths_m[1] = 1e-9
        for t = 2:num_total_days
            expected_daily_deaths_m[t] = sum([daily_cases_pred_m[τ] * πₘ[t - τ] for τ = 1:(t - 1)])
        end
    end

    ### Stan-equivalent ###
    # for(m in 1:M){
    #     for(i in EpidemicStart[m]:N[m]){
    #         deaths[i,m] ~ neg_binomial_2(E_deaths[i,m],phi);
    #     }
    # }
    for m = 1:num_countries
        expected_daily_deaths_m = expected_daily_deaths[m]
        for t = epidemic_start[m]:num_obs_countries[m]
            deaths[m][t] ~ NegativeBinomial2(expected_daily_deaths_m[t], ϕ)
        end
    end

    return daily_cases_pred, expected_daily_deaths, Rₜ
end
