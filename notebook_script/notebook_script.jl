using DrWatson

quickactivate(@__DIR__)

projectdir()

# ENV["PYTHON"] = "$(ENV['HOME'])/.local/bin/python"

# using Pkg
# Pkg.build("PyCall")

using Covid19
using Memoization, ReverseDiff, Turing, Bijectors, Optim, ForwardDiff
Turing.setrdcache(false)
Turing.setadbackend(:reversediff)
Turing.setadbackend(:forwarddiff)

using Random
using RData
import DrWatson: datadir

Random.seed!(1);

datadir() = projectdir("data", "imperial-report13")
datadir(s...) = projectdir("data", "imperial-report13", s...)

rdata = load(datadir("processed.rds"));

# Convert some misparsed fields
rdata["covariate4"] = Matrix(rdata["covariate4"]);

rdata["N2"] = Int(rdata["N2"]);
rdata["N0"] = Int(rdata["N0"]);

rdata["EpidemicStart"] = Int.(rdata["EpidemicStart"]);

rdata["cases"] = Int.(rdata["cases"]);
rdata["deaths"] = Int.(rdata["deaths"]);

rdata["x"] = []

#sum(data.num_obs_countries)

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

# TODO: not sure if either of the below is the correct ordering of the names!
# names_covariates = (
#     "Schools + Universities",
#     "Self-isolating if ill",
#     "Public events",
#     "Lockdown",
#     "Social distancing encouraged"
# )
names_covariates = ("schools_universities", "self_isolating_if_ill", "public_events", "any", "lockdown", "social_distancing_encouraged")

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
    "N2" => "num_total_days", "EpidemicStart" => "epidemic_start"
)

# Add some type-information to arrays and replace `-1` with `missing` (as `-1` is supposed to represent, well, missing data)
d["deaths"] = Int.(d["deaths"])
# d["deaths"] = replace(d["deaths"], -1 => missing)
d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix

d["cases"] = Int.(d["cases"])
# d["cases"] = replace(d["cases"], -1 => missing)
d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix

d["num_countries"] = Int(d["num_countries"])
d["num_total_days"] = Int(d["num_total_days"])
d["num_impute"] = Int(d["num_impute"])
d["num_obs_countries"] = Int.(d["num_obs_countries"])
d["epidemic_start"] = Int.(d["epidemic_start"])
d["population"] = Int.(d["population"])

d["π"] = collect(eachcol(d["π"])) # convert into Array of arrays instead of matrix

# Convert to `Matrix` if some are `DataFrame`
covariates = [Matrix(d["covariate$(i)"]) for i = 1:6]
# Array of matrices, with each element in array corresponding to (time, features)-matrix for a country
covariates = [hcat([covariates[i][:, m] for i = 1:6]...) for m = 1:d["num_countries"]]

data = (; (k => d[String(k)] for k in [:num_countries, :num_impute, :num_obs_countries, :num_total_days, :cases, :deaths, :π, :epidemic_start, :population, :serial_intervals])...)
data = merge(data, (covariates = covariates, ));

data.num_countries


m = ImperialReport13.model_v2_vectorized_multithreaded(
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
    data.serial_intervals
);

chain_prior = sample(m, Turing.Inference.PriorSampler(), 100);

params = (
    warmup = 100,
    steps = 200,
    model = "imperial-report13-v2-vectorized-reversediff"
)

function get_nlogp(model)
    # Construct a trace struct
    vi = Turing.VarInfo(model)

    # Define a function to optimize.
    function nlogp(x)
        spl = Turing.SampleFromPrior()
        new_vi = Turing.VarInfo(vi, spl, x)
        model(new_vi, spl)
        -Turing.getlogp(new_vi)
    end
    nlogp_g(g, x) = g .= ForwardDiff.gradient(nlogp, x)

    return nlogp, nlogp_g
end
nlogp, nlogp_g = get_nlogp(m)

x0 = Turing.VarInfo(m)[Turing.SampleFromPrior()]
lb = fill(2e-6, length(x0))
ub = fill(Inf, length(x0))
result = optimize(nlogp, nlogp_g, lb, ub, x0, Fminbox(GradientDescent()), Optim.Options(iterations = 100))
init_theta = result.minimizer

chain = sample(m, NUTS(params.warmup, 0.95; max_depth=10), params.steps + params.warmup, init_theta=init_theta);
write(savename("chains", params, "jls"), chain)

"""
    generated_quantities(m::Turing.Model, c::Turing.MCMCChains.Chains)

Executes `m` for each of the samples in `c` and returns an array of the values returned by the `m` for each sample.

## Examples
Often you might have additional quantities computed inside the model that you want to inspect, e.g.
```julia
@model demo(x) = begin
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()

    return interesting_quantity(θ, x)
end

m = demo(data)
chain = sample(m, alg, n)

# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples from the posterior/`chain`:
generated_quantities(m, chain)
```
"""
function generated_quantities(m::Turing.Model, c::Turing.MCMCChains.Chains)
    varinfo = Turing.DynamicPPL.VarInfo(m)

    map(1:length(c)) do i
        Turing.DynamicPPL._setval!(varinfo, c[i])
        m(varinfo)
    end
end


using Plots, StatsPlots
pyplot()

function plot_confidence!(p::Plots.Plot, data; label="", kwargs...)
    intervals = [0.025, 0.25, 0.5, 0.75, 0.975]
    
    qs = [quantile(v, intervals) for v in eachrow(data)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )
    plot!(mq, ribbon=(mq - llq, uuq - mq), linewidth=0, label="$label (95% quantiles)", kwargs...)
    plot!(mq, ribbon=(mq - lq, uq - mq), linewidth=0, label="$label (50% quantiles)", kwargs...)

    return p
end

plot_confidence(data; kwargs...) = plot_confidence!(plot(), data; kwargs...)

function country_prediction_plot(country_idx, e_deaths_country, Rₜ_country, predictions_country)
    p1 = bar(replace(data.deaths[country_idx], missing => 0.), label="$(countries[country_idx])")
    title!("Daily deaths")
    vline!([data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([data.num_obs_countries[country_idx]], label="end of observations", linewidth=2)

    p2 = plot_confidence(e_deaths_country; label = "$(countries[country_idx])")
    title!("Daily deaths (pred)")
    bar!(replace(data.deaths[country_idx], missing => 0.), label="$(countries[country_idx]) (observed)", alpha=0.5)

    p3 = plot_confidence(Rₜ_country; label = "$(countries[country_idx])")
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.covariates[country_idx])))
        if c_time !== nothing
            # c_name = names(covariates)[2:end][c_idx]
            c_name = names_covariates[c_idx]
            if !(c_name == "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!("Rₜ")

    # p3 = bar(replace(data.cases[country_idx], missing => -1.), label="$(countries[country_idx])")
    # title!("Daily cases")

    p4 = plot_confidence(predictions_country; label = "$(countries[country_idx])")
    title!("Daily cases (pred)")
    bar!(replace(data.cases[country_idx], missing => 0.), label="$(countries[country_idx]) (observed)", alpha=0.5)

    vals = cumsum(e_deaths_country; dims = 1)
    p5 = plot_confidence(vals; label = "$(countries[country_idx])")
    plot!(cumsum(data.deaths[country_idx]), label="observed")
    title!("Deaths (pred)")

    vals = cumsum(predictions_country; dims = 1)
    p6 = plot_confidence(vals; label = "$(countries[country_idx])")
    plot!(cumsum(data.cases[country_idx]), label="observed")
    title!("Cases (pred)")

    plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200))
end

# Compute the "generated quantities" for the PRIOR
res = generated_quantities(m, chain_prior)
prediction_prior = [x[1] for x in res];
expected_deaths_prior = [x[2] for x in res];
Rₜ_prior = [x[3] for x in res];

# Compute the "generated quantities" from the POSTERIOR
res = generated_quantities(m, chain)
prediction_chain = [x[1] for x in res];
expected_deaths_chain = [x[2] for x in res];
Rₜ_chain = [x[3] for x in res];

# HACK: eeehm this could be nicer:)
function country_prior_prediction_plot(country_idx)
    e_deaths_country = hcat([expected_deaths_prior[t][country_idx] for t = 1:length(chain_prior)]...)
    Rₜ_country = hcat([Rₜ_prior[t][country_idx] for t = 1:length(chain_prior)]...)
    predictions_country = hcat([prediction_prior[t][country_idx] for t = 1:length(chain_prior)]...)

    return country_prediction_plot(country_idx, e_deaths_country, Rₜ_country, predictions_country)
end

function country_posterior_prediction_plot(country_idx)
    e_deaths_country = hcat([expected_deaths_chain[t][country_idx] for t = 1:length(chain)]...)
    Rₜ_country = hcat([Rₜ_chain[t][country_idx] for t = 1:length(chain)]...)
    predictions_country = hcat([prediction_chain[t][country_idx] for t = 1:length(chain)]...)

    return country_prediction_plot(country_idx, e_deaths_country, Rₜ_country, predictions_country)
end

country_posterior_prediction_plot(7)

country_posterior_prediction_plot(7)

country_prior_prediction_plot(5)

using PyCall

using PyCall: pyimport
pystan = pyimport("pystan")

using LibGit2

imperialdir() = projectdir("external", "covid19model")
imperialdir(args...) = projectdir("external", "covid19model", args...)

if !ispath(imperialdir())
    mkpath(projectdir("external"))
    LibGit2.clone("https://github.com/ImperialCollegeLondon/covid19model.git", imperialdir())
end

sm = pystan.StanModel(file=imperialdir("stan-models", "base.stan"))

keys(rdata)

d = Dict([(k, rdata[k]) for k in keys(rdata)]) # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`

fit_stan(n_iters=2_000) = sm.sampling(
    data=d, iter=n_iters, chains=1, warmup=100, algorithm="NUTS", 
    control=Dict(
        "adapt_delta" => 0.95,
        "max_treedepth" => 10
    )
)
f = fit_stan(300)

la = f.extract(permuted=true)

stan_chain = Chains(la["y"], ["y[$i]" for i = 1:size(la["y"], 2)])

chain[:y]

p1 = plot(stan_chain[:y]);

p2 = plot(chain[:y]);

plot(p1, p2, layout = (1, 2))

country_idx = 1

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 2

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 3

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 4

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 5

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 6

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

country_idx = 7

vals = la["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_chain[i][country_idx] for i = 1:length(chain)]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

stan_prior_model_str = """
data {
  int <lower=1> M; // number of countries
  int <lower=1> N0; // number of days for which to impute infections
  int<lower=1> N[M]; // days of observed data for country m. each entry must be <= N2
  int<lower=1> N2; // days of observed data + # of days to forecast
  int cases[N2,M]; // reported cases
  int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  matrix[N2, M] f; // h * s
  matrix[N2, M] covariate1;
  matrix[N2, M] covariate2;
  matrix[N2, M] covariate3;
  matrix[N2, M] covariate4;
  matrix[N2, M] covariate5;
  matrix[N2, M] covariate6;
  int EpidemicStart[M];
  real pop[M];
  real SI[N2]; // fixed pre-calculated SI using emprical data from Neil
}

generated quantities {
  real<lower=0> mu[M]; // intercept for Rt
  real<lower=0> alpha_hier[6]; // sudo parameter for the hier term for alpha
  real<lower=0> kappa;
  real<lower=0> y[M];
  real<lower=0> phi;
  real<lower=0> tau;
  real<lower=0> ifr_noise[M];

  real alpha[6];
  matrix[N2, M] prediction = rep_matrix(0,N2,M);
  matrix[N2, M] E_deaths  = rep_matrix(0,N2,M);
  matrix[N2, M] Rt = rep_matrix(0,N2,M);
  matrix[N2, M] Rt_adj = Rt;

  matrix[N2, M] prediction0 = rep_matrix(0,N2,M);
  matrix[N2, M] E_deaths0  = rep_matrix(0,N2,M);

  tau = exponential_rng(0.03);
  for (m in 1:M){
    y[m] = exponential_rng(1/tau);
  }
  phi = fabs(normal_rng(0,5));
  kappa = fabs(normal_rng(0,0.5));
  for (m in 1:M) mu[m] = fabs(normal_rng(3.28, kappa)); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
  for (i in 1:6) alpha_hier[i] = gamma_rng(.1667,1);
  for (m in 1:M) ifr_noise[m] = fabs(normal_rng(1,0.1));
    
  {
    matrix[N2,M] cumm_sum = rep_matrix(0,N2,M);
    for(i in 1:6){
      alpha[i] = alpha_hier[i] - ( log(1.05) / 6.0 );
    }
    for (m in 1:M){
      for (i in 2:N0){
        cumm_sum[i,m] = cumm_sum[i-1,m] + y[m]; 
      }
      prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        
      Rt[,m] = mu[m] * exp( covariate1[,m] * (-alpha[1]) + covariate2[,m] * (-alpha[2]) +
                            covariate3[,m] * (-alpha[3]) + covariate4[,m] * (-alpha[4]) + covariate5[,m] * (-alpha[5]) + 
                            covariate6[,m] * (-alpha[6]) );
      Rt_adj[1:N0,m] = Rt[1:N0,m];
      for (i in (N0+1):N2) {
        real convolution=0;
        for(j in 1:(i-1)) {
          convolution += prediction[j, m] * SI[i-j];
        }
        cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
        Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
        prediction[i, m] = Rt_adj[i,m] * convolution;
      }
        
      E_deaths[1, m]= 1e-15 * prediction[1,m];
      for (i in 2:N2){
        for(j in 1:(i-1)){
          E_deaths[i,m] += prediction[j,m] * f[i-j,m] * ifr_noise[m];
        }
      }
    }
  }

  {
    matrix[N2,M] cumm_sum0 = rep_matrix(0,N2,M);
    for (m in 1:M){
      for (i in 2:N0){
        cumm_sum0[i,m] = cumm_sum0[i-1,m] + y[m]; 
      }
      prediction0[1:N0,m] = rep_vector(y[m],N0); 
      for (i in (N0+1):N2) {
        real convolution0 = 0;
        for(j in 1:(i-1)) {
          convolution0 += prediction0[j, m] * SI[i-j]; 
        }
        cumm_sum0[i,m] = cumm_sum0[i-1,m] + prediction0[i-1,m];
        prediction0[i, m] =  ((pop[m]-cumm_sum0[i,m]) / pop[m]) * mu[m] * convolution0;
      }
        
      E_deaths0[1, m] = uniform_rng(1e-16, 1e-15);
      for (i in 2:N2){
        for(j in 1:(i-1)){
          E_deaths0[i,m] += prediction0[j,m] * f[i-j,m] * ifr_noise[m];
        }
      }
    }
  }
}

""";

sm_prior = pystan.StanModel(model_code=stan_prior_model_str)

fit_stan_prior(n_iters=2_000) = sm_prior.sampling(
    # I believe `refresh` ensures that we 
    data=d, iter=n_iters, chains=1, warmup=0, algorithm="Fixed_param", refresh=n_iters
)

f_prior = fit_stan_prior(10000)

la_prior = f_prior.extract(permuted=true)

mean_Rₜ_stan = mean(la_prior["Rt"]; dims = 1)[1, :, :]

res = generated_quantities(m, chain_prior)
prediction_prior = [x[1] for x in res];
expected_deaths_prior = [x[2] for x in res];
Rₜ_prior = [x[3] for x in res];

mean_Rₜ_turing = mean([hcat(Rₜ_prior[i]...) for i = 1:length(Rₜ_prior)])

p1 = plot(mean_Rₜ_stan)
title!("Rₜ (Stan)")

p2 = plot(mean_Rₜ_turing)
title!("Rₜ (Turing)")

plot(p1, p2, size = (900, 300))
ylims!(0, 5)

stan_prior_chain = Chains(la_prior["y"], ["y[$i]" for i = 1:size(la_prior["y"], 2)])

p1 = plot(stan_prior_chain[:y]);

p2 = plot(chain_prior[:y]);

plot(p1, p2, layout = (1, 2))

country_idx = 7

hcat([Rₜ_prior[i][country_idx] for i = 1:10_000]...)'

la_prior["Rt"][:, :, country_idx]

vals = la_prior["Rt"][:, :, country_idx]'
p1 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Stan)")

vals = hcat([Rₜ_prior[i][country_idx] for i = 1:10_000]...)
p2 = plot_confidence(vals; label = "$(countries[country_idx])")
title!("Prior (Turing)")

plot(p1, p2; layout = (1, 2), size = (900, 300))

stan_chain_alpha_hier = Chains(la_prior["alpha_hier"], ["α_hier[$i]" for i = 1:6])

p1 = plot(stan_chain_alpha_hier[:α_hier])
title!("Stan")
p2 = plot(chain_prior[:α_hier])
title!("Turing")

plot(p1, p2, layout = (1, 2))

varinfo = Turing.VarInfo(m);
@code_warntype m.f(m, varinfo, Turing.DynamicPPL.SampleFromPrior(), Turing.DynamicPPL.DefaultContext())
