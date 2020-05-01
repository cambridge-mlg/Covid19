using DrWatson
quickactivate(@__DIR__)

using Pkg
Pkg.instantiate()

# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19

# Some other packages we'll need
using Random, Dates, Turing, Bijectors

using Base.Threads
nthreads()

begin
    dist = product_distribution(fill(truncated(Normal(3.28, 10), 0, Inf), 14))
    b = bijector(dist)
    x = rand(dist)

    # stupid check to make sure that we have the correct versions of Bijectors.jl used WITHIN Turing.jl
    @assert (invlink(dist, b(x)) ≈ Turing.invlink(dist, b(x))) && (Turing.invlink(dist, b(x)) ≈ x)
end

begin
    @assert length(unique(rand(Turing.filldist(Gamma(.1667, 1), 6)))) > 1
end

using Pkg
Pkg.status()

import DrWatson: datadir

datadir() = projectdir("data", "imperial-report13")
datadir(s...) = projectdir("data", "imperial-report13", s...)

using RData

rdata_full = load(datadir("processed_new.rds"))
rdata = rdata_full["stan_data"];

keys(rdata_full)

country_to_dates = d = Dict([(k, rdata_full["dates"][k]) for k in keys(rdata_full["dates"])])

# Convert some misparsed fields
rdata["N2"] = Int(rdata["N2"]);
rdata["N0"] = Int(rdata["N0"]);

rdata["EpidemicStart"] = Int.(rdata["EpidemicStart"]);

rdata["cases"] = Int.(rdata["cases"]);
rdata["deaths"] = Int.(rdata["deaths"]);

# Stan will fail if these are `nothing` so we make them empty arrays
rdata["x"] = []
rdata["features"] = []

countries = (
  "Denmark",
  "Italy",
  "Germany",
  "Spain",
  "United_Kingdom",
  "France",
  "Norway",
  "Belgium",
  "Austria", 
  "Sweden",
  "Switzerland",
  "Greece",
  "Portugal",
  "Netherlands"
)
num_countries = length(countries)

names_covariates = ("schools_universities", "self_isolating_if_ill", "public_events", "any", "lockdown", "social_distancing_encouraged")
lockdown_index = findfirst(==("lockdown"), names_covariates)


function rename!(d, names::Pair...)
    # check that keys are not yet present before updating `d`
    for k_new in values.(names)
        @assert k_new ∉ keys(d) "$(k_new) already in dictionary"
    end
    
    for (k_old, k_new) in names
        d[k_new] = pop!(d, k_old)
    end
    return d
end

# `rdata` is a `DictOfVector` so we convert to a simple `Dict` for simplicity
d = Dict([(k, rdata[k]) for k in keys(rdata)]) # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`

# Rename some columns
rename!(
    d,
    "f" => "π", "SI" => "serial_intervals", "pop" => "population",
    "M" => "num_countries", "N0" => "num_impute", "N" => "num_obs_countries",
    "N2" => "num_total_days", "EpidemicStart" => "epidemic_start",
    "X" => "covariates", "P" => "num_covariates"
)

# Add some type-information to arrays and replace `-1` with `missing` (as `-1` is supposed to represent, well, missing data)
d["deaths"] = Int.(d["deaths"])
# d["deaths"] = replace(d["deaths"], -1 => missing)
d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix

d["cases"] = Int.(d["cases"])
# d["cases"] = replace(d["cases"], -1 => missing)
d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix

d["num_covariates"] = Int(d["num_covariates"])
d["num_countries"] = Int(d["num_countries"])
d["num_total_days"] = Int(d["num_total_days"])
d["num_impute"] = Int(d["num_impute"])
d["num_obs_countries"] = Int.(d["num_obs_countries"])
d["epidemic_start"] = Int.(d["epidemic_start"])
d["population"] = Int.(d["population"])

d["π"] = collect(eachcol(d["π"])) # convert into Array of arrays instead of matrix

# Convert 3D array into Array{Matrix}
covariates = [rdata["X"][m, :, :] for m = 1:num_countries]

data = (; (k => d[String(k)] for k in [:num_countries, :num_impute, :num_obs_countries, :num_total_days, :cases, :deaths, :π, :epidemic_start, :population, :serial_intervals])...)
data = merge(data, (covariates = covariates, ));

data.num_countries

uk_index = findfirst(==("United_Kingdom"), countries)

@model model_v2(
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

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rₜ = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rₜ_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # country-specific parameters
        πₘ = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rₜ_m = Rₜ[m]
        Rₜ_adj_m = Rₜ_adj[m]
        
        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        cases_pred_m[1] = zero(eltype(cases_pred_m))
        cases_pred_m[2:num_impute] = cumsum(fill(y[m], num_impute - 1)) .+ cases_pred_m[1]

        expected_daily_cases_m[1:num_impute] .= y[m]
        
        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rₜ_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # adjusts for portion of pop that are susceptible
        Rₜ_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rₜ_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            Rₜ_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rₜ_m[t] # adjusts for portion of pop that are susceptible
            expected_daily_cases_m[t] = Rₜ_adj_m[t] * sum([expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1)])
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum([expected_daily_cases_m[τ] * πₘ[t - τ] * ifr_noise[m] for τ = 1:(t - 1)])
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
    _varinfo.logp[] += sum(logps)

    # These are the "generated quantities" that we're interested in analysing
    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rₜ = Rₜ,
        Rₜ_adjusted = Rₜ_adj
    )
end;

model_def = model_v2

# Model instantance used to for inference
m_no_pred = model_def(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    false # <= DON'T predict
);

# Model instance used for prediction
m = model_def(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= predict
);

res = m();
res.expected_daily_cases[uk_index]

using Plots, StatsPlots
pyplot()

# Ehh, this can be made nicer...
function country_prediction_plot(country_idx, predictions_country::AbstractMatrix, e_deaths_country::AbstractMatrix, Rₜ_country::AbstractMatrix; normalize_pop::Bool = false)
    pop = data.population[country_idx]
    num_total_days = length(data.deaths[country_idx])
    num_observed_days = data.num_obs_countries[country_idx]

    country_name = countries[country_idx]
    start_date = first(country_to_dates[country_name])
    dates = cumsum(fill(Day(1), data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = preproc(replace(data.deaths[country_idx][1:num_observed_days], missing => 0.))
    daily_cases = preproc(replace(data.cases[country_idx][1:num_observed_days], missing => 0.))
    
    p1 = bar(daily_deaths, label="$(country_name)", xaxis=false)
    title!("Observed daily deaths")
    vline!([data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([data.num_obs_countries[country_idx]], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_country); label = "$(countries[country_idx])")
    title!("Expected daily deaths (pred)")
    bar!(daily_deaths, label="$(countries[country_idx]) (observed)", alpha=0.5)

    p3 = plot(; xaxis=false)
    plot_confidence_timeseries!(p3, Rₜ_country; label = "$(countries[country_idx])")
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.covariates[country_idx])))
        if c_time !== nothing
            # c_name = names(covariates)[2:end][c_idx]
            c_name = names_covariates[c_idx]
            if (c_name != "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!("Rₜ")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rₜ_country)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    # p3 = bar(replace(data.cases[country_idx], missing => -1.), label="$(countries[country_idx])")
    # title!("Daily cases")

    p4 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_country); label = "$(countries[country_idx])")
    title!("Expected daily cases (pred)")
    bar!(daily_cases, label="$(countries[country_idx]) (observed)", alpha=0.5)

    vals = cumsum(e_deaths_country; dims = 1)
    p5 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p5, preproc(vals); label = "$(countries[country_idx])")
    plot!(cumsum(daily_deaths), label="observed", color=:red)
    title!("Expected deaths (pred)")

    vals = cumsum(predictions_country; dims = 1)
    p6 = plot(; legend = :topleft)
    plot_confidence_timeseries!(p6, preproc(vals); label = "$(countries[country_idx])")
    plot!(daily_cases, label="observed", color=:red)
    title!("Expected cases (pred)")

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200), sharex=true)
    xticks!(1:3:num_total_days, date_strings[1:3:end], xrotation=45)

    return p
end
                                        
function country_prediction_plot(country_idx, cases, e_deaths, Rₜ; kwargs...)
    n = length(cases)
    e_deaths_country = hcat([e_deaths[t][country_idx] for t = 1:n]...)
    Rₜ_country = hcat([Rₜ[t][country_idx] for t = 1:n]...)
    predictions_country = hcat([cases[t][country_idx] for t = 1:n]...)

    return country_prediction_plot(country_idx, predictions_country, e_deaths_country, Rₜ_country; kwargs...)
end

chain_prior = sample(m, Turing.Inference.PriorSampler(), 1_000);

plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)

# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rₜ_prior, Rₜ_adj_prior = generated_prior;

country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rₜ_prior)

country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rₜ_adj_prior)

data.population[uk_index]

parameters = (
    warmup = 1000,
    steps = 3000
);

chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)

filenames = [
    projectdir("out", s) for s in readdir(projectdir("out"))
    if occursin(savename(parameters), s) && occursin("seed", s)
]
length(filenames)

chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # concatenate them
chains_posterior = chains_posterior[1:3:end] # <= thin so we're left with 1000 samples

# rename some variables to make the chain compatible with new model where we've changed a variable name from μ₀ → μ
chains_posterior = set_names(chains_posterior, Dict{String, String}(["μ₀[$i]" => "μ[$i]" for i = 1:length(names(chains_posterior[:μ₀]))]))

plot(chains_posterior[[:κ, :ϕ, :τ]]; α = .5, linewidth=1.5)

# Compute generated quantities for the chains pooled together
pooled_chains = MCMCChains.pool_chain(chains_posterior)
generated_posterior = vectup2tupvec(generated_quantities(m, pooled_chains));

daily_cases_posterior, daily_deaths_posterior, Rₜ_posterior, Rₜ_adj_posterior = generated_posterior;

country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rₜ_posterior)

country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rₜ_adj_posterior)

country_idx = 0

country_idx += 1
country_prediction_plot(country_idx, daily_cases_prior, daily_deaths_prior, Rₜ_adj_prior)

country_prediction_plot(country_idx, daily_cases_posterior, daily_deaths_posterior, Rₜ_adj_posterior)

# Get the index of schools and univerities closing
schools_universities_closed_index = findfirst(==("schools_universities"), names_covariates)
# Time-series for UK
data.covariates[uk_index][:, schools_universities_closed_index]

"""
    zero_covariates(xs::AbstractMatrix{<:Real}; remove=[], keep=[])

Allows you to zero out covariates if the name of the covariate is in `remove` or NOT zero out those in `keep`.
Note that only `remove` xor `keep` can be non-empty.

Useful when instantiating counter-factual models, as it allows one to remove/keep a subset of the covariates.
"""
zero_covariates(xs::AbstractMatrix{<:Real}; kwargs...) = zero_covariates(xs, names_covariates; kwargs...)
function zero_covariates(xs::AbstractMatrix{<:Real}, names_covariates; remove=[], keep=[])
    @assert (isempty(remove) || isempty(keep)) "only `remove` or `keep` can be non-empty"
    
    if isempty(keep)
        return hcat([
            (names_covariates[i] ∈ remove ? zeros(eltype(c), length(c)) : c) 
            for (i, c) in enumerate(eachcol(xs))
        ]...)
    else
        return hcat([
        (names_covariates[i] ∈ keep ? c : zeros(eltype(c), length(c))) 
        for (i, c) in enumerate(eachcol(xs))
        ]...)
    end
end

# What happens if we don't do anything?
m_counterfactual = model_def(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zeros(size(c)) for c in data.covariates], # <= remove ALL covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_counterfactual, Rₜ_adj_counterfactual = generated_counterfactual;
country_prediction_plot(5, daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_adj_counterfactual)

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zero_covariates(c; remove = ["lockdown", "schools_universities"]) for c in data.covariates], # <= remove covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_counterfactual, Rₜ_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_adj_counterfactual)

lift_lockdown_time = 70

new_covariates = [copy(c) for c in data.covariates] # <= going to do inplace manipulations so we copy
for covariates_m ∈ new_covariates
    covariates_m[lift_lockdown_time:end, lockdown_index] .= 0
end

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    new_covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_counterfactual, Rₜ_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rₜ_adj_counterfactual)
