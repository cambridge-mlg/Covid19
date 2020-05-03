# Replication of https://github.com/ImperialCollegeLondon/covid19model
using Turing
using Distributions, StatsBase
using ArgCheck
using FillArrays

using Base.Threads

import Turing: filldist

######################
## Model definition ##
######################
@model model_v1(
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
    τ ~ Exponential(1 / 0.03) # Exponential has inverse parameterization of the one in Stan

    y = fill(TV(undef, num_impute), num_countries)
    for m = 1:num_countries
        y[m] .~ Exponential(τ)
    end
    # y ~ arraydist(fill(Exponential(τ), (num_countries, num_impute)))
    ϕ ~ truncated(Normal(0, 5), 1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
    κ ~ truncated(Normal(0, 0.5), 1e-6, 100) # In Stan they don't make this truncated, but specify that `κ ≥ 0` and so it will be transformed
    # κ ~ Turing.Bijectors.transformed(Normal(0, 0.5), Turing.Bijectors.Exp{0}())
    μ ~ product_distribution(fill(truncated(Normal(2.4, κ), 0, Inf), num_countries))
    # μ ~ Turing.Bijectors.transformed(product_distribution(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries))), Turing.Bijectors.Exp{1}())
    α ~ product_distribution(fill(Gamma(.5, 1), num_covariates))

    # Transforming variables
    daily_cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    Rₜ = TV[TV(undef, num_total_days) for _ in 1:num_countries]

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
        Rₜ_m .= μ[m] * exp.(covariates[m] * (- α))

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


@model model_v2_old(
    num_countries,     # [Int] num. of countries
    num_impute,        # [Int] num. of days for which to impute infections
    num_obs_countries, # [Vector{Int}] days of observed data for country `m`; each entry must be ≤ N2
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
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
    @argcheck all([size(covariates[m]) == (num_total_days, num_covariates) for m = 1:num_countries])
    @argcheck size(epidemic_start) == (num_countries, )
    @argcheck size(population, ) == (num_countries, )
    @argcheck size(serial_intervals) == (num_total_days, )

    # Sample variables
    τ ~ Exponential(1 / 0.03) # Exponential has inverse parameterization of the one in Stan
    y ~ product_distribution(fill(Exponential(τ), num_countries))
    ϕ ~ truncated(Normal(0, 5), 1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
    κ ~ truncated(Normal(0, 0.5), 1e-6, 100) # In Stan they don't make this truncated, but specify that `κ ≥ 0` and so it will be transformed
    # κ ~ Turing.Bijectors.transformed(Normal(0, 0.5), Turing.Bijectors.Exp{0}())

    # HACK: often ran into numerical issues when transforming from `truncated` in this case...
    # μ ~ Turing.Bijectors.transformed(product_distribution(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries))), Turing.Bijectors.Exp{1}())
    # μ ~ product_distribution(fill(truncated(Normal(3.28, κ), 1e-6, Inf), num_countries))
    μ₀ ~ product_distribution(fill(Normal(3.28, κ), num_countries))
    μ = abs.(μ₀)

    α_hier ~ product_distribution(fill(Gamma(.1667, 1), num_covariates))
    α = α_hier .- log(1.05) / 6.

    # TODO: fixed ifr noise over time? Seems a bit strange, no?
    ifr_noise ~ product_distribution(fill(truncated(Normal(1., 0.1), 1e-6, 1000), num_countries))

    # Transforming variables
    daily_cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    Rₜ = TV[TV(undef, num_total_days) for _ in 1:num_countries]

    for m = 1:num_countries
        # country-specific parameters
        πₘ = π[m]
        pop_m = population[m]
        daily_cases_pred_m = daily_cases_pred[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rₜ_m = Rₜ[m]

        # Imputation of `num_impute` first days (since we're looking quite far into the past)

        ### Stan-equivalent ###
        # for (i in 2:N0){
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + y[m]; 
        # }
        # prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        # Rt[,m] = mu[m] * exp( covariate1[,m] * (-alpha[1]) + covariate2[,m] * (-alpha[2]) +
        #   covariate3[,m] * (-alpha[3]) + covariate4[,m] * (-alpha[4]) + covariate5[,m] * (-alpha[5]) + 
        #   covariate6[,m] * (-alpha[6]) );
        #   Rt_adj[1:N0,m] = Rt[1:N0,m];
        cases_pred_m[1] = zero(eltype(cases_pred_m))
        for t = 2:num_impute
            cases_pred_m[t] = cases_pred_m[t - 1] + y[m]
        end
        daily_cases_pred_m[1:num_impute] .= y[m]
        Rₜ_m .= μ[m] * exp.(covariates[m] * (-α))

        ### Stan-equivalent ###        
        # for (i in (N0+1):N2) {
        #   real convolution=0;
        #   for(j in 1:(i-1)) {
        #     convolution += prediction[j, m] * SI[i-j];
        #   }
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
        #   Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
        #   prediction[i, m] = Rt_adj[i,m] * convolution;
        # }
        for t = (num_impute + 1):num_total_days
            # cases_pred = sum(daily_cases_pred_m[1:(t - 1)])
            cases_pred_m[t] = cases_pred_m[t - 1] + daily_cases_pred_m[t - 1]

            Rₜ_adj = (max(pop_m - cases_pred_m[t], eps(cases_pred_m[t])) / pop_m) * Rₜ_m[t] # adjusts for portion of pop that are susceptible
            daily_cases_pred_m[t] = Rₜ_adj * sum([daily_cases_pred_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1)])
        end

        ### Stan-equivalent ###
        # E_deaths[1, m]= 1e-9;
        # for (i in 2:N2){
        #     E_deaths[i,m]= 0;
        #     for(j in 1:(i-1)){
        #         E_deaths[i,m] += daily_cases_pred[j,m]*f[i-j,m];
        #     }
        # }
        expected_daily_deaths_m[1] = 1e-15 * daily_cases_pred_m[1]
        for t = 2:num_total_days
            expected_daily_deaths_m[t] = sum([daily_cases_pred_m[τ] * πₘ[t - τ] * ifr_noise[m] for τ = 1:(t - 1)])
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

@model model_v2_vectorized(
    num_countries,     # [Int] num. of countries
    num_impute,        # [Int] num. of days for which to impute infections
    num_obs_countries, # [Vector{Int}] days of observed data for country `m`; each entry must be ≤ N2
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    ::Type{TV} = Vector{Float64}
) where {TV} = begin
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)

    # Sample variables
    τ ~ Exponential(1 / 0.03) # Exponential has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
    κ ~ truncated(Normal(0, 0.5), 1e-6, 100) # In Stan they don't make this truncated, but specify that `κ ≥ 0` and so it will be transformed
    # κ ~ Turing.Bijectors.transformed(Normal(0, 0.5), Turing.Bijectors.Exp{0}())

    # HACK: often ran into numerical issues when transforming from `truncated` in this case...
    # μ ~ Turing.Bijectors.transformed(product_distribution(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries))), Turing.Bijectors.Exp{1}())
    # μ ~ product_distribution(fill(truncated(Normal(3.28, κ), 1e-6, Inf), num_countries))
    μ₀ ~ filldist(Normal(3.28, κ), num_countries)
    μ = abs.(μ₀)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    # TODO: fixed ifr noise over time? Seems a bit strange, no?
    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 1e-6, 1000), num_countries)

    # Transforming variables
    daily_cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    cases_pred = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, num_total_days) for _ in 1:num_countries]
    Rₜ = TV[TV(undef, num_total_days) for _ in 1:num_countries]

    for m = 1:num_countries
        # country-specific parameters
        πₘ = π[m]
        pop_m = population[m]
        daily_cases_pred_m = daily_cases_pred[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rₜ_m = Rₜ[m]

        # Imputation of `num_impute` first days (since we're looking quite far into the past)

        ### Stan-equivalent ###
        # for (i in 2:N0){
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + y[m]; 
        # }
        # prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        # Rt[,m] = mu[m] * exp( covariate1[,m] * (-alpha[1]) + covariate2[,m] * (-alpha[2]) +
        #   covariate3[,m] * (-alpha[3]) + covariate4[,m] * (-alpha[4]) + covariate5[,m] * (-alpha[5]) + 
        #   covariate6[,m] * (-alpha[6]) );
        #   Rt_adj[1:N0,m] = Rt[1:N0,m];
        cases_pred_m[1] = zero(eltype(cases_pred_m))
        cases_pred_m[2:num_impute] = cumsum(fill(y[m], num_impute - 1)) .+ cases_pred_m[1]

        daily_cases_pred_m[1:num_impute] .= y[m]
        Rₜ_m .= μ[m] * exp.(covariates[m] * (-α))

        ### Stan-equivalent ###        
        # for (i in (N0+1):N2) {
        #   real convolution=0;
        #   for(j in 1:(i-1)) {
        #     convolution += prediction[j, m] * SI[i-j];
        #   }
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
        #   Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
        #   prediction[i, m] = Rt_adj[i,m] * convolution;
        # }
        for t = (num_impute + 1):num_total_days
            # cases_pred = sum(daily_cases_pred_m[1:(t - 1)])
            cases_pred_m[t] = cases_pred_m[t - 1] + daily_cases_pred_m[t - 1]

            Rₜ_adj = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rₜ_m[t] # adjusts for portion of pop that are susceptible
            daily_cases_pred_m[t] = Rₜ_adj * sum([daily_cases_pred_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1)])
        end

        ### Stan-equivalent ###
        # E_deaths[1, m]= 1e-9;
        # for (i in 2:N2){
        #     E_deaths[i,m]= 0;
        #     for(j in 1:(i-1)){
        #         E_deaths[i,m] += daily_cases_pred[j,m]*f[i-j,m];
        #     }
        # }
        expected_daily_deaths_m[1] = 1e-15 * daily_cases_pred_m[1]
        for t = 2:num_total_days
            expected_daily_deaths_m[t] = sum([daily_cases_pred_m[τ] * πₘ[t - τ] * ifr_noise[m] for τ = 1:(t - 1)])
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
        ts = epidemic_start[m]:num_obs_countries[m]
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end

    return daily_cases_pred, expected_daily_deaths, Rₜ
end


@model model_v2_vectorized_multithreaded(
    num_countries,     # [Int] num. of countries
    num_impute,        # [Int] num. of days for which to impute infections
    num_obs_countries, # [Vector{Int}] days of observed data for country `m`; each entry must be ≤ N2
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV} = begin
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)

    # Sample variables
    τ ~ Exponential(1 / 0.03) # Exponential has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
    κ ~ truncated(Normal(0, 0.5), 1e-6, 100) # In Stan they don't make this truncated, but specify that `κ ≥ 0` and so it will be transformed
    # κ ~ Turing.Bijectors.transformed(Normal(0, 0.5), Turing.Bijectors.Exp{0}())

    # HACK: often ran into numerical issues when transforming from `truncated` in this case...
    # μ ~ Turing.Bijectors.transformed(product_distribution(Normal.(2.4 .* ones(num_countries), κ .* ones(num_countries))), Turing.Bijectors.Exp{1}())
    # μ ~ product_distribution(fill(truncated(Normal(3.28, κ), 1e-6, Inf), num_countries))
    μ₀ ~ filldist(Normal(3.28, κ), num_countries)
    μ = abs.(μ₀)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    # TODO: fixed ifr noise over time? Seems a bit strange, no?
    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 1e-6, 1000), num_countries)
    
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Transforming variables
    daily_cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rₜ = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rₜ_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # NOTE: making this threaded shaved off ~1/3 of the runtime
    @threads for m = 1:num_countries
        # country-specific parameters
        πₘ = π[m]
        pop_m = population[m]
        daily_cases_pred_m = daily_cases_pred[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rₜ_m = Rₜ[m]
        Rₜ_adj_m = Rₜ_adj[m]
        
        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` first days (since we're looking quite far into the past)

        ### Stan-equivalent ###
        # for (i in 2:N0){
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + y[m]; 
        # }
        # prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        # Rt[,m] = mu[m] * exp( covariate1[,m] * (-alpha[1]) + covariate2[,m] * (-alpha[2]) +
        #   covariate3[,m] * (-alpha[3]) + covariate4[,m] * (-alpha[4]) + covariate5[,m] * (-alpha[5]) + 
        #   covariate6[,m] * (-alpha[6]) );
        #   Rt_adj[1:N0,m] = Rt[1:N0,m];
        cases_pred_m[1] = zero(eltype(cases_pred_m))
        cases_pred_m[2:num_impute] = cumsum(fill(y[m], num_impute - 1)) .+ cases_pred_m[1]

        daily_cases_pred_m[1:num_impute] .= y[m]
        
        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps
        Rₜ_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # adjusts for portion of pop that are susceptible
        Rₜ_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rₜ_m[1:num_impute]

        ### Stan-equivalent ###        
        # for (i in (N0+1):N2) {
        #   real convolution=0;
        #   for(j in 1:(i-1)) {
        #     convolution += prediction[j, m] * SI[i-j];
        #   }
        #   cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
        #   Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
        #   prediction[i, m] = Rt_adj[i,m] * convolution;
        # }
        for t = (num_impute + 1):last_time_step
            cases_pred_m[t] = cases_pred_m[t - 1] + daily_cases_pred_m[t - 1]

            Rₜ_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rₜ_m[t] # adjusts for portion of pop that are susceptible
            daily_cases_pred_m[t] = Rₜ_adj_m[t] * sum([daily_cases_pred_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1)])
        end

        ### Stan-equivalent ###
        # E_deaths[1, m]= 1e-9;
        # for (i in 2:N2){
        #     E_deaths[i,m]= 0;
        #     for(j in 1:(i-1)){
        #         E_deaths[i,m] += daily_cases_pred[j,m]*f[i-j,m];
        #     }
        # }
        expected_daily_deaths_m[1] = 1e-15 * daily_cases_pred_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum([daily_cases_pred_m[τ] * πₘ[t - τ] * ifr_noise[m] for τ = 1:(t - 1)])
        end
    end

    ### Stan-equivalent ###
    # for(m in 1:M){
    #     for(i in EpidemicStart[m]:N[m]){
    #         deaths[i,m] ~ neg_binomial_2(E_deaths[i,m],phi);
    #     }
    # }

    # NOTE: making this threaded didn't seem to speed things up much
    logps = TV(undef, num_countries)
    @threads for m = 1:num_countries
        expected_daily_deaths_m = expected_daily_deaths[m]
        ts = epidemic_start[m]:num_obs_countries[m]
        # deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
        logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
    end
    _varinfo.logp[] += sum(logps)

    return (
        daily_cases_pred = daily_cases_pred,
        expected_daily_deaths = expected_daily_deaths,
        Rₜ = Rₜ,
        Rₜ_adjusted = Rₜ_adj
    )
end

# This is the most up-to-date one
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # Country-specific parameters
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # Adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t]

            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum(expected_daily_cases_m[τ] * π_m[t - τ] * ifr_noise[m] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    logps = TV(undef, num_countries)
    @threads for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
    end
    Turing.acclogp!(_varinfo, sum(logps))

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
