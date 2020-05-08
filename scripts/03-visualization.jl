#########################################################################
# This file is tangled from notebooks/03-Imperial-Report13-analysis.org #
#########################################################################

using DrWatson

# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19

# Some other packages we'll need
using Random, Dates, Turing, Bijectors

using Base.Threads
nthreads()

outdir() = projectdir("out")
outdir(args...) = projectdir("out", args...)

data = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"));

country_to_dates = data.country_to_dates

countries = data.countries;
num_countries = length(data.countries);
covariate_names = data.covariate_names;

lockdown_index = findfirst(==("lockdown"), covariate_names)

# Need to pass arguments to `pystan` as a `Dict` with different names, so we have one instance tailored for `Stan` and one for `Turing.jl`
stan_data = data.stan_data;
turing_data = data.turing_data;

num_countries

uk_index = findfirst(==("United_Kingdom"), countries)

model_def = ImperialReport13.model;

# Model instantance used to for inference
m_no_pred = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    turing_data.covariates,
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    lockdown_index,
    false # <= DON'T predict
);

# Model instance used for prediction
m = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    turing_data.covariates,
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    lockdown_index,
    true # <= predict
);

res = m();
res.expected_daily_cases[uk_index]

chain_prior = sample(m, Turing.Inference.Prior(), 4_000);

using Plots, StatsPlots, LaTeXStrings

pyplot()

plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)

# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior; # <= tuple of `Vector{<:Vector{<:Vector}}`

ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_prior, daily_deaths_prior, Rt_prior; main_title = "(prior)")

ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_prior, daily_deaths_prior, Rt_adj_prior; main_title = "(prior)")

turing_data.population[uk_index]

parameters = (
    warmup = 1000,
    steps = 3000
);

chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)

using Serialization

stan_chain_fname = first([s for s in readdir(outdir()) if occursin("stan", s)])
la = open(io -> deserialize(io), outdir(stan_chain_fname), "r")

Covid19.rename!(
    la,
    "alpha" => "α",
    "alpha_hier" => "α_hier",
    "kappa" => "κ",
    "gamma" => "γ",
    "mu" => "μ",
    "phi" => "ϕ",
    "tau" => "τ"
)

# Extract a subset of the variables, since we don't want everything in a `Chains` object
la_subset = Dict(
    k => la[k] for k in 
    ["y", "κ", "α_hier", "ϕ", "τ", "ifr_noise", "μ", "γ", "lockdown"]
)

filenames = [
    relpath(outdir(s)) for s in readdir(outdir())
    if occursin(savename(parameters), s) && occursin("seed", s)
]
length(filenames)

chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # Read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # Concatenate them
chains_posterior = chains_posterior[1:3:end] # <= Thin so we're left with 1000 samples

plot(chains_posterior[[:κ, :ϕ, :τ]]; α = .5, linewidth=1.5)

# Compute generated quantities for the chains pooled together
pooled_chains = MCMCChains.pool_chain(chains_posterior)
generated_posterior = vectup2tupvec(generated_quantities(m, pooled_chains));

daily_cases_posterior, daily_deaths_posterior, Rt_posterior, Rt_adj_posterior = generated_posterior;

# Convert `Vector{<:Vector{<:Vector}}` into `Array{<:Real, 3}` with shape `(num_countries, num_days, num_samples)`
daily_cases_posterior_arr = permutedims(arrarrarr2arr(daily_cases_posterior), (2, 3, 1));
daily_deaths_posterior_arr = permutedims(arrarrarr2arr(daily_deaths_posterior), (2, 3, 1));
Rt_posterior_arr = permutedims(arrarrarr2arr(Rt_posterior), (2, 3, 1));
Rt_adj_posterior_arr = permutedims(arrarrarr2arr(Rt_adj_posterior), (2, 3, 1));

ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_posterior; main_title = "(posterior)")

ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior; main_title = "(posterior)")

plotlyjs();

ImperialReport13.countries_prediction_plot(data, Rt_posterior_arr; size = (800, 300))
title!("Rt (95% intervals)")

ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_posterior_arr; dims = 2); size = (800, 300))
title!("Expected cases (95% intervals)")

ImperialReport13.countries_prediction_plot(data, cumsum(daily_deaths_posterior_arr; dims = 2); size = (800, 300))
title!("Expected deaths (95% intervals)")

ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_posterior_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected cases / population (95% intervals)")

ImperialReport13.countries_prediction_plot(data, cumsum(daily_deaths_posterior_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected deaths / population (95% intervals)")

pyplot()

pyplot()

country_idx = 0

country_idx += 1
ImperialReport13.country_prediction_plot(data, country_idx, daily_cases_prior, daily_deaths_prior, Rt_adj_prior; main_title = "(prior)")

ImperialReport13.country_prediction_plot(data, country_idx, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior; main_title = "(posterior)")

# Get the index of schools and univerities closing
schools_universities_closed_index = findfirst(==("schools_universities"), covariate_names)
# Time-series for UK
turing_data.covariates[uk_index][:, schools_universities_closed_index]

"""
    zero_covariates(xs::AbstractMatrix{<:Real}; remove=[], keep=[])

Allows you to zero out covariates if the name of the covariate is in `remove` or NOT zero out those in `keep`.
Note that only `remove` xor `keep` can be non-empty.

Useful when instantiating counter-factual models, as it allows one to remove/keep a subset of the covariates.
"""
zero_covariates(xs::AbstractMatrix{<:Real}; kwargs...) = zero_covariates(xs, covariate_names; kwargs...)
function zero_covariates(xs::AbstractMatrix{<:Real}, covariate_names; remove=[], keep=[])
    @assert (isempty(remove) || isempty(keep)) "only `remove` or `keep` can be non-empty"
    
    if isempty(keep)
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (covariate_names[i] ∈ remove ? zeros(eltype(c), length(c)) : c)
        end
    else
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (covariate_names[i] ∈ keep ? c : zeros(eltype(c), length(c))) 
        end
    end
end

# What happens if we don't do anything?
m_counterfactual = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    [zeros(size(c)) for c in turing_data.covariates], # <= remove ALL covariates
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;

# Convert `Vector{<:Vector{<:Vector}}` into `Array{<:Real, 3}` with shape `(num_countries, num_days, num_samples)`
daily_cases_counterfactual_arr = permutedims(arrarrarr2arr(daily_cases_counterfactual), (2, 3, 1));
daily_deaths_counterfactual_arr = permutedims(arrarrarr2arr(daily_deaths_counterfactual), (2, 3, 1));
Rt_counterfactual_arr = permutedims(arrarrarr2arr(Rt_counterfactual), (2, 3, 1));
Rt_adj_counterfactual_arr = permutedims(arrarrarr2arr(Rt_adj_counterfactual), (2, 3, 1));

# plot
ImperialReport13.country_prediction_plot(data, 5, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)

plotlyjs()
ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_counterfactual_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected cases / population when no intervention (95% intervals)")

pyplot()

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    [zero_covariates(c; remove = ["lockdown", "schools_universities"]) for c in turing_data.covariates], # <= remove covariates
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)

lift_lockdown_time = 75

new_covariates = [copy(c) for c in turing_data.covariates] # <= going to do inplace manipulations so we copy
for covariates_m ∈ new_covariates
    covariates_m[lift_lockdown_time:end, lockdown_index] .= 0
end

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    new_covariates,
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
