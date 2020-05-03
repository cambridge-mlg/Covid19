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

outdir() = projectdir("out")
outdir(args...) = projectdir("out", args...)

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

# Can deal with ragged arrays, so we can shave off unobserved data (future) which are just filled with -1
data = merge(
    data,
    (cases = [data.cases[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries],
     deaths = [data.deaths[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries])
);

data.num_countries

uk_index = findfirst(==("United_Kingdom"), countries)

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

model_def = model_v2;

# Model instantance used to for inference
m_no_pred = model_def(
    data.num_impute,
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
    data.num_impute,
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
function country_prediction_plot(country_idx, predictions_country::AbstractMatrix, e_deaths_country::AbstractMatrix, Rt_country::AbstractMatrix; normalize_pop::Bool = false)
    pop = data.population[country_idx]
    num_total_days = data.num_total_days
    num_observed_days = length(data.cases[country_idx])

    country_name = countries[country_idx]
    start_date = first(country_to_dates[country_name])
    dates = cumsum(fill(Day(1), data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = data.deaths[country_idx]
    daily_cases = data.cases[country_idx]
    
    p1 = plot(; xaxis = false, legend = :topleft)
    bar!(preproc(daily_deaths), label="Observed daily deaths")
    title!(replace(country_name, "_" => " "))
    vline!([data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([num_observed_days], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_country); label = "Expected daily deaths")
    # title!("Expected daily deaths (pred)")
    bar!(preproc(daily_deaths), label="Recorded daily deaths (observed)", alpha=0.5)

    p3 = plot(; legend = :bottomleft, xaxis=false)
    plot_confidence_timeseries!(p3, Rt_country; no_label = true)
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.covariates[country_idx])))
        if c_time !== nothing
            c_name = names_covariates[c_idx]
            if (c_name != "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!("Rt")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rt_country)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    p4 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_country); label = "Expected daily cases")
    # title!("Expected daily cases (pred)")
    bar!(preproc(daily_cases), label="Recorded daily cases (observed)", alpha=0.5)

    vals = preproc(cumsum(e_deaths_country; dims = 1))
    p5 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p5, vals; label = "Expected deaths")
    plot!(preproc(cumsum(daily_deaths)), label="Recorded deaths (observed)", color=:red)
    # title!("Expected deaths (pred)")

    vals = preproc(cumsum(predictions_country; dims = 1))
    p6 = plot(; legend = :topleft)
    plot_confidence_timeseries!(p6, vals; label = "Expected cases")
    plot!(preproc(daily_cases), label="Recorded cases (observed)", color=:red)
    # title!("Expected cases (pred)")

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200), sharex=true)
    xticks!(1:3:num_total_days, date_strings[1:3:end], xrotation=45)

    return p
end
                                        
function country_prediction_plot(country_idx, cases, e_deaths, Rt; kwargs...)
    n = length(cases)
    e_deaths_country = hcat([e_deaths[t][country_idx] for t = 1:n]...)
    Rt_country = hcat([Rt[t][country_idx] for t = 1:n]...)
    predictions_country = hcat([cases[t][country_idx] for t = 1:n]...)

    return country_prediction_plot(country_idx, predictions_country, e_deaths_country, Rt_country; kwargs...)
end

chain_prior = sample(m, Turing.Inference.PriorSampler(), 1_000);

plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)

# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior;

country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_prior)

country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_adj_prior)

data.population[uk_index]

parameters = (
    warmup = 1000,
    steps = 3000
);

chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)

using PyCall

using PyCall: pyimport
pystan = pyimport("pystan");

model_str = raw"""
data {
  int <lower=1> M; // number of countries
  int <lower=1> P; // number of covariates
  int <lower=1> N0; // number of days for which to impute infections
  int<lower=1> N[M]; // days of observed data for country m. each entry must be <= N2
  int<lower=1> N2; // days of observed data + # of days to forecast
  int cases[N2,M]; // reported cases
  int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  matrix[N2, M] f; // h * s
  matrix[N2, P] X[M]; // features matrix
  int EpidemicStart[M];
  real pop[M];
  real SI[N2]; // fixed pre-calculated SI using emprical data from Neil
}

transformed data {
  vector[N2] SI_rev; // SI in reverse order
  vector[N2] f_rev[M]; // f in reversed order
  
  for(i in 1:N2)
    SI_rev[i] = SI[N2-i+1];
    
  for(m in 1:M){
    for(i in 1:N2) {
     f_rev[m, i] = f[N2-i+1,m];
    }
  }
}


parameters {
  real<lower=0> mu[M]; // intercept for Rt
  real<lower=0> alpha_hier[P]; // sudo parameter for the hier term for alpha
  real<lower=0> gamma;
  vector[M] lockdown;
  real<lower=0> kappa;
  real<lower=0> y[M];
  real<lower=0> phi;
  real<lower=0> tau;
  real <lower=0> ifr_noise[M];
}

transformed parameters {
    vector[P] alpha;
    matrix[N2, M] prediction = rep_matrix(0,N2,M);
    matrix[N2, M] E_deaths  = rep_matrix(0,N2,M);
    matrix[N2, M] Rt = rep_matrix(0,N2,M);
    matrix[N2, M] Rt_adj = Rt;
    
    {
      matrix[N2,M] cumm_sum = rep_matrix(0,N2,M);
      for(i in 1:P){
        alpha[i] = alpha_hier[i] - ( log(1.05) / 6.0 );
      }
      for (m in 1:M){
        prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        cumm_sum[2:N0,m] = cumulative_sum(prediction[2:N0,m]);
        
        Rt[,m] = mu[m] * exp(-X[m] * alpha - X[m][,5] * lockdown[m]);
        Rt_adj[1:N0,m] = Rt[1:N0,m];
        for (i in (N0+1):N2) {
          real convolution = dot_product(sub_col(prediction, 1, m, i-1), tail(SI_rev, i-1));
          cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
          Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
          prediction[i, m] = Rt_adj[i,m] * convolution;
        }
        E_deaths[1, m]= 1e-15 * prediction[1,m];
        for (i in 2:N2){
          E_deaths[i,m] = ifr_noise[m] * dot_product(sub_col(prediction, 1, m, i-1), tail(f_rev[m], i-1));
        }
      }
    }
}
model {
  tau ~ exponential(0.03);
  for (m in 1:M){
      y[m] ~ exponential(1/tau);
  }
  gamma ~ normal(0,.2);
  lockdown ~ normal(0,gamma);
  phi ~ normal(0,5);
  kappa ~ normal(0,0.5);
  mu ~ normal(3.28, kappa); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
  alpha_hier ~ gamma(.1667,1);
  ifr_noise ~ normal(1,0.1);
  for(m in 1:M){
    deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
   }
}

generated quantities {
    matrix[N2, M] prediction0 = rep_matrix(0,N2,M);
    matrix[N2, M] E_deaths0  = rep_matrix(0,N2,M);
    
    {
      matrix[N2,M] cumm_sum0 = rep_matrix(0,N2,M);
      for (m in 1:M){
         for (i in 2:N0){
          cumm_sum0[i,m] = cumm_sum0[i-1,m] + y[m]; 
        }
        prediction0[1:N0,m] = rep_vector(y[m],N0); 
        for (i in (N0+1):N2) {
          real convolution0 = dot_product(sub_col(prediction0, 1, m, i-1), tail(SI_rev, i-1));
          cumm_sum0[i,m] = cumm_sum0[i-1,m] + prediction0[i-1,m];
          prediction0[i, m] = ((pop[m]-cumm_sum0[i,m]) / pop[m]) * mu[m] * convolution0;
        }
        E_deaths0[1, m]= 1e-15 * prediction0[1,m];
        for (i in 2:N2){
          E_deaths0[i,m] = ifr_noise[m] * dot_product(sub_col(prediction0, 1, m, i-1), tail(f_rev[m], i-1));
        }
      }
    }
}
"""

d = Dict([(k, rdata[k]) for k in keys(rdata)]); # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`

sm = pystan.StanModel(model_code=model_str)

fit_stan(n_iters=300, warmup=100) = sm.sampling(
    data=d, iter=n_iters, chains=1, warmup=warmup, algorithm="NUTS", 
    control=Dict(
        "adapt_delta" => 0.95,
        "max_treedepth" => 10
    )
)
f = fit_stan(parameters.steps + parameters.warmup, parameters.warmup)

la = f.extract(permuted=true)

using Serialization

stan_chain_fname = first([s for s in readdir(outdir()) if occursin("stan", s)])
la = open(io -> deserialize(io), outdir(stan_chain_fname), "r")

rename!(
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

function MCMCChains._cat(::Val{3}, c1::Chains, args::Chains...)
    # check inputs
    rng = range(c1)
    # (OR not, he he he)
    # all(c -> range(c) == rng, args) || throw(ArgumentError("chain ranges differ"))
    nms = names(c1)
    all(c -> names(c) == nms, args) || throw(ArgumentError("chain names differ"))

    # concatenate all chains
    data = mapreduce(c -> c.value.data, (x, y) -> cat(x, y; dims = 3), args;
                     init = c1.value.data)
    value = MCMCChains.AxisArray(data; iter = rng, var = nms, chain = 1:size(data, 3))

    return Chains(value, missing, c1.name_map, c1.info)
end

stan_chain = Chains(la_subset); # <= results in all chains being concatenated together so we need to manually "separate" them

steps_per_chain = parameters.steps
num_chains = Int(length(stan_chain) // steps_per_chain)

stan_chains = [stan_chain[1 + (i - 1) * steps_per_chain:i * steps_per_chain] for i = 1:num_chains];
stan_chains = chainscat(stan_chains...);
stan_chains = stan_chains[1:3:end] # thin

filenames = [
    relpath(outdir(s)) for s in readdir(outdir())
    if occursin(savename(parameters), s) && occursin("seed", s)
]
filenames = [
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=1_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=2_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=3_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=4_steps=3000_warmup=1000_with_lockdown=true.jls",
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

country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_posterior)

country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior)

country_idx = 0

country_idx += 1
country_prediction_plot(country_idx, daily_cases_prior, daily_deaths_prior, Rt_adj_prior)

country_prediction_plot(country_idx, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior)

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
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (names_covariates[i] ∈ remove ? zeros(eltype(c), length(c)) : c)
        end
    else
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (names_covariates[i] ∈ keep ? c : zeros(eltype(c), length(c))) 
        end
    end
end

# What happens if we don't do anything?
m_counterfactual = model_def(
    data.num_impute,
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
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(5, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
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
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)

lift_lockdown_time = 75

new_covariates = [copy(c) for c in data.covariates] # <= going to do inplace manipulations so we copy
for covariates_m ∈ new_covariates
    covariates_m[lift_lockdown_time:end, lockdown_index] .= 0
end

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
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
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
