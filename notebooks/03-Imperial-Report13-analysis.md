We, i.e. the [TuringLang team](https://turing.ml/dev/team/), are currently exploring cooperation with other researchers in attempt to help with the ongoing crisis. As preparation for this and to get our feet wet, we decided it would be useful to do a replication study of the [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/). We figured it might be useful for the public, in particular other researchers working on the same or similar models, to see the results of this analysis, and thus decided to make it available here.

We want to emphasize that you should look to the original paper rather than this post for developments and analysis of the model. We are not aiming to make any claims about the validity or the implications of the model and refer to [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/) for details on the model itself. This post's purpose is only to add tiny bit of validation to the *inference* performed in the paper by obtaining the same results using a different probabilistic programming language (PPL) and to explore whether or not `Turing.jl` can be useful for researchers working on these problems.

All code and inference results shown in this post can be found [here](https://github.com/TuringLang/Covid19).


# Setup

This is all assuming that you're in the project directory of [`Covid19.jl`](https://github.com/TuringLang/Covid19), a small package where we gather most of our ongoing work.

In the project we use [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) which provides a lot of convenient functionality for, well, working with a project. The below code will activate the `Covid19.jl` project to ensure that we're using correct versions of all the dependencies, i.e. code is reproducible. It's basically just doing `] activate` but with a few bells and whistles that we'll use later.

```julia
using DrWatson
quickactivate(@__DIR__)
```

With the project activated, we can import the `Covid19.jl` package:

```julia
# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19
```

```julia
# Some other packages we'll need
using Random, Dates, Turing, Bijectors
```

And we'll be using the new [multithreading functionality in Julia](https://julialang.org/blog/2019/07/multithreading/) to speed things up, so we need the `Base.Threads` package.

```julia
using Base.Threads
nthreads()
```

In the Github project you will find a `Manifest.toml`. This means that if you're working directory is the project directory, and you do `julia --project` followed by `] instantiate` you will have *exactly* the same enviroment as we had when performing this analysis.

<details><summary>Overloading some functionality in <code>DrWatson.jl</code></summary>

As mentioned `DrWatson.jl` provides a lot of convenience functions for working in a project, and one of them is `projectdir(args...)` which will resolve join the `args` to the absolute path of the project directory. We add a `outdir` method for resolving the paths for some of the files we will be loading.

```julia
outdir() = projectdir("out")
outdir(args...) = projectdir("out", args...)
```

</details>


# Data

To ensure consistency with the original implementation from the paper, the `data` (or input) is obtained using the `base.r` script from the [original repository (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096), with the exception of renaming some variables. Hence the input should be identical to the input outlined in the paper, and we defer a thorough description of the inputs to their excellent [techical report (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf). Here's a short summary of the inputs to the model:

-   `cases` and `deaths` which are *daily* recorded cases and deaths, respectively
-   `covariates` which refers to different interventions taken by the country, e.g. closing schools and universities
-   `epidemic_start` which refers to be "start of the epidemic" in a specific country, which is defined as 30 days prior to the first 10 cumulative recorded deaths
-   `serial_intervals` which are pre-computed serial intervals → see their technical report for their estimation technique
-   `π` (see model definition below)

The procssed data (using their `base.r`) is stored in `out/imperial-report13/processed.rds` in the project. To load this RDS we simply do:

```julia
data = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"));
```

```julia
country_to_dates = data.country_to_dates
```

For convenience we can extract the some of the loaded data into global variables:

```julia
countries = data.countries;
num_countries = length(data.countries);
covariate_names = data.covariate_names;

lockdown_index = findfirst(==("lockdown"), covariate_names)

# Need to pass arguments to `pystan` as a `Dict` with different names, so we have one instance tailored for `Stan` and one for `Turing.jl`
stan_data = data.stan_data;
turing_data = data.turing_data;
```

```julia
num_countries
```

Because it's a bit much to visualize 14 countries at each step, we're going to use UK as an example throughout.

```julia
uk_index = findfirst(==("United_Kingdom"), countries)
```

<span style="color: red;">TODO: remove the text below when we've re-run models with updated data!!!</span>

It's worth noting that the data user here is not quite up-to-date for UK because on <span class="timestamp-wrapper"><span class="timestamp">&lt;2020-04-30 to.&gt; </span></span> deaths from care- and nursing-homes were included, and this has then been smoothed over the past days by the [ECDC](https://www.ecdc.europa.eu/en), which is the data source. Thus if you compare the prediction of the model to real numbers, it's likely that the real numbers will be a bit higher than what the model predicts with the current data.


# Model

For a thorough description of the model and the assumptions that have gone into it, we recommend looking at the [original paper](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/) or their very nice [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf). The model described here is the one corresponding to the technical report linked. The link points to the correct commit ID and so should be consistent with this post despite potential changes to the "official" model made in the future.

For the sake of exposition, we present a compact version of the model here:

\begin{align}
  \tau & \sim \mathrm{Exponential}(1 / 0.03) \\
  y_m & \sim \mathrm{Exponential}(\tau) \quad & \text{for} \quad m = 1, \dots, M \\
  \kappa & \sim \mathcal{N}^{ + }(0, 0.5) \\
  \mu_m & \sim \mathcal{N}^{ + }(3.28, \kappa) \quad & \text{for} \quad m = 1, \dots, M \\
  \gamma & \sim \mathcal{N}^{ + }(0, 0.2) \\
  \beta_m & \sim \mathcal{N}(0, \gamma) \quad & \text{for} \quad m = 1, \dots, M \\
  \tilde{\alpha}_k &\sim \mathrm{Gamma}(0.1667, 1) \quad & \text{for} \quad k = 1, \dots, K \\
  \alpha_k &= \tilde{\alpha}_k - \frac{\log(1.05)}{6} \quad & \text{for} \quad  k = 1, \dots, K \\
  R_{t, m} &= \mu_m \exp(- \beta_m x_{k_{\text{ld}}} - \sum_{k=1}^{K} \alpha_k x_k) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T  \\
  \tilde{R}_{t, m} &= 1 - \frac{1}{p_m} \sum_{\tau = 1}^{t - 1} c_{\tau, m}  \quad & \text{for} \quad m = 1, \dots, M, \ t = T_{\text{impute}} + 1, \dots, T \\
  c_{t, m} &= y_m \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T_{\text{impute}} \\
  c_{t, m} &= \tilde{R}_{t, m} \sum_{\tau = 1}^{t - 1} c_{\tau, m} s_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = T_{\text{impute}} + 1, \dots, T \\
  \varepsilon_m^{\text{ifr}} &\sim \mathcal{N}(1, 0.1)^{ + } \quad & \text{for} \quad m = 1, \dots, M \\
  \mathrm{ifr}_m^{ * } &\sim \mathrm{ifr}_m \cdot \varepsilon_m^{\text{ifr}} \quad & \text{for} \quad m = 1, \dots, M \\
  d_{t, m} &= \mathrm{ifr}_m^{ * } \sum_{\tau=1}^{t - 1} c_{\tau, m} \pi_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T \\
  \phi  & \sim \mathcal{N}^{ + }(0, 5) \\
  D_{t, m} &\sim \mathrm{NegativeBinomial}(d_{t, m}, \phi) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T 
\end{align}

where

-   it's assumed that seeding of new infections begins 30 days before the day after a country has cumulative observed 10 deaths
-   \(M\) denotes the number of countries
-   \(T\) the total number of time-steps
-   \(T_{\text{impute}}\) the time steps to *impute* values for; the first 6 of the 30 days we impute the number, and then we simulate the rest
-   \(\alpha_k\) denotes the weights for the k-th intervention/covariate
-   \(\beta_m\) denotes the weight for the `lockdown` intervention (whose index we denote by \(k_{\text{ld}}\))
    -   Note that there is also a \(\alpha_{k_{\text{ld}}}\) which is shared between all the \(M\) countries
    -   In contrast, the \(\beta_m\) weight is local to the country with index \(m\)
    -   This is a sort of way to try and deal with the fact that `lockdown` means different things in different countries, e.g. `lockdown` in UK is much more severe than "lockdown" in Norway.
-   \(\mu_m\) represents the \(R_0\) value for country \(m\) (i.e. \(R_t\) without any interventions)
-   \(R_{t, m}\) denotes the **reproduction number** at time \(t\) for country \(m\)
-   \(\tilde{R}_{t, m}\) denotes the **adjusted reproduction number** at time \(t\) for country \(m\), *adjusted* in the sense that it's rescaled wrt. what proportion of the population is susceptible for infection (assuming infected people cannot get the virus again within the near future)
-   \(p_{m}\) denotes the **total/initial population** for country \(m\)
-   \(\mathrm{ifr}_m\) denotes the **infection-fatality ratio** for country \(m\), and \(\mathrm{ifr}_m^{ * }\) the *adjusted* infection-fatality ratio (see paper)
-   \(\varepsilon_m^{\text{ifr}}\) denotes the noise for the multiplicative noise for the \(\mathrm{ifr}_m^{ * }\)
-   \(\pi\) denotes the **time from infection to death** and is assumed to be a sum of two independent random times: the incubation period (*infection-to-onset*) and time between onset of symptoms and death (*onset-to-death*):
    
    \begin{equation*}
    \pi \sim \mathrm{Gamma}(5.1, 0.86) + \mathrm{Gamma}(18.8, 0.45)
    \end{equation*}
    
    where in this case the \(\mathrm{Gamma}\) is parameterized by its mean and coefficient of variation. In the model, this is a *precomputed* quantity and not something to be inferred, though in an ideal world this would also be included in the model.
-   \(\pi_t\) then denotes a discretized version of the PDF for \(\pi\). The reasoning behind the discretization is that if we assume \(d_m(t)\) to be a continuous random variable denoting the death-rate at any time \(t\), then it would be given by
    
    \begin{equation*}
    d_m(t) = \mathrm{ifr}_m^{ * } \int_0^t c_m(\tau) \pi(t - \tau) dt
    \end{equation*}
    
    i.e. the convolution of the number of cases observed at time time \(\tau\), \(c_m(\tau)\), and the *probability* of death at prior to time \(t\) for the new cases observed at time \(\tau\), \(\pi(t - \tau)\) (assuming stationarity of \(\pi(t)\)). Thus, \(c_m(\tau) \pi(t - \tau)\) can be interpreted as the portion people who got the virus at time \(\tau\) have died at time \(t\) (or rather, have died after having the virus for \(t - \tau\) time, with \(t > \tau\)). Discretizing then results in the above model.
-   \(s_t\) denotes the **serial intervals**, i.e. the time between successive cases in a chain of transmission, also a precomputed quantity
-   \(c_{t, m}\) denotes the **expected daily cases** at time \(t\) for country \(m\)
-   \(C_{t, m}\) denotes the **cumulative cases** prior to time \(t\) for country \(m\)
-   \(d_{t, m}\) denotes the **expected daily deaths** at time \(t\) for country \(m\)
-   \(D_{t, m}\) denotes the **daily deaths** at time \(t\) for country \(m\) (in our case, this is the **likelihood**); note that here we're using the mean and variance coefficient parameterization of \(\mathrm{NegativeBinomial}\)

To see the reasoning for the choices of distributions and parameters for the priors, see the either the paper or the [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf).


## Code

In `Turing.jl`, a "sample"-statement is defined by `x ~ Distribution`. Therefore, the priors used in the model can be written within a Turing.jl `Model` as:

```julia
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
```

The `filldist` function in the above snippet is a function used to construct a `Distribution` from which we can obtain i.i.d. samples from a univariate distribution using vectorization.

And the full model is defined:

```julia
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
    for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
```

Two things worth noting is the use of the `TV` variable to instantiate some internal variables and the use of `@threads`.

As you can see in the arguments for the model, `TV` refers to a *type* and will be recognized as such by the `@model` macro when transforming the model code. This is used to ensure *type-stability* of the model.

<details><summary>More detailed explanation of <code>TV</code></summary>

A default execution of the model will then use `TV` as `Vector{Float64}`, thus making statements like `TV(undef, n)` result in a `Vector{Float64}` with `undef` (uninitialized) values and length `n`. But in case we use a sampler that requires automatic differentiation (AD), TV will be will be replaced with the AD-type corresponding to a `Vector`, e.g. `TrackedVector` in the case of [`Tracker.jl`](https://github.com/FluxML/Tracker.jl).

</details>

The use of `@threads` means that inside each execution of the `Model`, this loop will be performed in parallel, where the number of threads are specified by the enviroment variable `JULIA_NUM_THREADS`. This is thanks to the really nice multithreading functionality [introduced in Julia 1.3](https://julialang.org/blog/2019/07/multithreading/) (and so this also requires Julia 1.3 or higher to run the code). Note that the inside the loop is independent of each other and each `m` will be seen by only one thread, hence it's threadsafe.

This model is basically identitical to the one defined in [stan-models/base.stan (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/blob/6ee3010a58a57cc14a16545ae897ca668b7c9096/stan-models/base.stan) with the exception of two points:

-   In this model we use `TruncatedNormal` for normally distributed variables which are positively constrained instead of sampling from a `Normal` and then taking the absolute value; these approaches are equivalent from a modelling perspective.
-   We've added the use of `max(pop_m - cases_pred_m[t], 0)` in computing the *adjusted* \(R_t\), `Rt_adj`, to ensure that in the case where the entire populations has died there, the adjusted \(R_t\) is set to 0, i.e. if everyone in the country passed away then there is no spread of the virus (this does not affect "correctness" of inference). <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>
-   The `cases` and `deaths` arguments are arrays of arrays instead of 3D arrays, therefore we don't need to fill the future days with `-1` as is done in the original model.


### Multithreaded observe

We can also make the `observe` statements parallel, but because the `~` is not (yet) threadsafe we unfortunately have to touch some of the internals of `Turing.jl`. But for observations it's very straight-forward: instead of observing by the following piece of code

```julia
for m = 1:num_countries
    # Extract the estimated expected daily deaths for country `m`
    expected_daily_deaths_m = expected_daily_deaths[m]
    # Extract time-steps for which we have observations
    ts = epidemic_start[m]:num_obs_countries[m]
    # Observe!
    deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
end
```

we can use the following

```julia
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
```

You can find the resulting model in the project.

<details><summary>Explanation of what we just did</summary>

It might be worth explaining a bit about what's going on here. First we should explain what the deal is with `_varinfo`. `_varinfo` is basically the object used internally in Turing to track the sampled variables and the log-pdf *for a particular evaluation* of the model, and so `acclogp!(_varinfo, lp)` will increment the log-pdf stored in `_varinfo` by `lp`. With that we can explain what happens to `~` inside the `@macro`. Using the old observe-snippet as an example, the `@model` macro replaces `~` with

```julia
acclogp!(_varinfo., logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts]))
```

But we're iterating through `m`, so this would not be thread-safe since you might be two threads attempting to mutate `_varinfo` simultaneously.<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> Therefore, since no threads sees the same `m`, delaying the accumulation to after having computed all the log-pdf in parallel leaves us with equivalent code that is threadsafe.

You can read more about the `@macro` and its internals [here](https://turing.ml/dev/docs/for-developers/compiler#model-macro-and-modelgen).

</details>


### Instantiating the model

We define an alias `model_def` so that if we want to try out a different model, there's only one point in the notebook which we need to change.

```julia
model_def = ImperialReport13.model_v2;
```

The input data have up to 30-40 days of unobserved future data which we might want to predict on. But during sampling we don't want to waste computation on sampling for the future for which we do not have any observations. Therefore we have an argument `predict::Bool` in the model which allows us to specify whether or not to generate future quantities.

```julia
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
```

```julia
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
```

Just to make sure everything is working, we can "evaluate" the model to obtain a sample from the prior:

```julia
res = m();
res.expected_daily_cases[uk_index]
```


# Visualization utilities

For visualisation we of course use [Plots.jl](https://github.com/JuliaPlots/Plots.jl), and in this case we're going to use the `pyplot` backend which uses Python's matplotlib under the hood.

```julia
chain_prior = sample(m, Turing.Inference.Prior(), 3_000);
```

```julia
using Plots, StatsPlots, LaTeXStrings
```

For the most part we will use the `PyPlot.jl` (which uses Python's `matplotlib` under the hood) backend for `Plots.jl`, but certain parts will be more useful to display using the `PlotlyJS.jl` (which uses `plotly.js` under the hood) backend.

```julia
pyplot()
```


# Prior

Before we do any inference it can be useful to inspect the *prior* distribution, in particular if you are working with a hierarchical model where the dependencies in the prior might lead to some unexpected behavior. In Turing.jl you can sample a chain from the prior using `sample`, much in the same way as you would sample from the posterior.

```julia
plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)
```

For the same reasons it can be very useful to inspect the *prior predictive* distribution.

```julia
# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior; # <= tuple of `Vector{<:Vector{<:Vector}}`
```

```julia
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_prior, daily_deaths_prior, Rt_prior; main_title = "(prior)")
```

And with the Rt *adjusted for remaining population*:

```julia
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_prior, daily_deaths_prior, Rt_adj_prior; main_title = "(prior)")
```

At this point it might be useful to remind ourselves of the total population of UK is:

```julia
turing_data.population[uk_index]
```

As we can see from the figures, the prior allows scenarios such as

-   *all* of the UK being infected
-   effects of interventions, e.g. `lockdown`, having a *negative* effect on `Rt` (in the sense that it can actually *increase* the spread of the virus); you can see this from the fact that the 95% confidence interval widens after one of the interventions

But at the same time, it's clear that a very sudden increase from 0% to 100% of the population being infected is almost impossible under the prior. All in all, the model prior seems a reasonable choice: it allows for extreme situations without putting too much probabilty "mass" on those, while still encoding some structure in the model.


# Posterior inference

```julia
parameters = (
    warmup = 1000,
    steps = 3000
);
```


## Inference


### Run

To perform inference for the model we would simply run the code below:

```julia
chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)
```

*But* unfortunately it takes quite a while to run. Performing inference using `NUTS` with `1000` steps for adaptation/warmup and `3000` sample steps takes ~2hrs on a 6-core computer with `JULIA_NUM_THREADS = 6`. And so we're instead going to load in the chains needed.

In contrast, `Stan` only takes roughly 1hr *on a single thread* using the base model from the repository. On a single thread `Turing.jl` is ~4-5X slower for this model, which is quite signficant.

This generally means that if you have a clear model in mind (or you're already very familiar with `Stan`), you probably want to use `Stan` for these kind of models. On the other hand, if you're in the process of heavily tweaking your model and need to be able to do `m = model(data...); m()` to check if it works, or you want more flexibility in what you can do with your model, e.g. discrete variables, `Turing.jl` might be a good option.

And this is an additional reason why we wanted to perform this replication study: we want `Turing.jl` to be *useful* and the way to check this is by applying `Turing.jl` to real-world problem rather than *just* benchmarks (though those are important too).

And regarding the performance difference, it really comes down to the difference in implementation of automatic differentation (AD). `Turing.jl` allows you to choose from the goto AD packages in Julia, e.g. in our runs we used [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), while `Stan` as a very, very fast AD implementation written exclusively for `Stan`. This difference becomes very clear in models such as this one where we have a lot of for-loops and recursive relationships (because this means that we can't easily vectorize). For-loops in Julia are generally blazingly fast, but with AD there's a bit of overhead. But that also means that in `Turing.jl` you have the ability to choose between different approaches to AD, i.e. forward-mode or reverse-mode, each with their different tradeoffs, and thus will benefit from potentially interesting future work, e.g. source-to-source AD using [Zygote.jl](https://github.com/FluxML/Zygote.jl).<sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup>

And one interesting tidbit is that you can very easily use `pystan` within `PyCall.jl` to sample from a `Stan` model, and then convert the results into a [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl). This has some nice implications:

-   we can use all the convenient posterior analysis tools available in MCMCChains.jl to analyze chains from `Stan`
-   we can use the `generated_quantities` method in this notebook to execute the `Turing.jl` `Model` on the samples obtain using `Stan`

This was quite useful for us to be able validate the results from `Turing.jl` against those from `Stan`, and made it very easy to check that indeed `Turing.jl` and `Stan` produce the same results. You can find examples of this in the [notebooks in our repository](https://github.com/TuringLang/Covid19).

<details><summary>Sampling from <code>Stan</code> in <code>Julia</code> using <code>pystan</code></summary>

First we import `PyCall`, allowing us to call Python code from within Julia.

```julia
using PyCall

using PyCall: pyimport
pystan = pyimport("pystan");
```

Then we define the Stan model as a string

```julia
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
```

Then we can compile the `Stan` model

```julia
sm = pystan.StanModel(model_code=model_str)
```

And finally fit:

```julia
fit_stan(n_iters=300, warmup=100) = sm.sampling(
    data=stan_data, iter=n_iters, chains=1, warmup=warmup, algorithm="NUTS", 
    control=Dict(
        "adapt_delta" => 0.95,
        "max_treedepth" => 10
    )
)
f = fit_stan(parameters.steps + parameters.warmup, parameters.warmup)
```

From the fit we can extract the inferred parameters:

```julia
la = f.extract(permuted=true)
```

Or, if you've done this before and saved the results using `Serialization.jl`, you can load the results:

```julia
using Serialization

stan_chain_fname = first([s for s in readdir(outdir()) if occursin("stan", s)])
la = open(io -> deserialize(io), outdir(stan_chain_fname), "r")
```

And if we want to compare it with the results from `Turing.jl` it can be convenient to rename some of the variables

```julia
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
```

```julia
# Extract a subset of the variables, since we don't want everything in a `Chains` object
la_subset = Dict(
    k => la[k] for k in 
    ["y", "κ", "α_hier", "ϕ", "τ", "ifr_noise", "μ", "γ", "lockdown"]
)
```

In `Covid19.jl` we've added a constructor for `MCMCChains.Chains` which takes a `Dict` as an argument, easily allowing us to convert the `la` from `Stan` into a `Chains` object.

```julia
# TODO: remove this and use `resetrange` after we've updated the dependencies
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
```

```julia
stan_chain = Chains(la_subset); # <= results in all chains being concatenated together so we need to manually "separate" them

steps_per_chain = parameters.steps
num_chains = Int(length(stan_chain) // steps_per_chain)

stan_chains = [stan_chain[1 + (i - 1) * steps_per_chain:i * steps_per_chain] for i = 1:num_chains];
stan_chains = chainscat(stan_chains...);
stan_chains = stan_chains[1:3:end] # thin
```

</details>


### Load

To download the resulting chains, you need to setup [`git lfs`](https://git-lfs.github.com/) and clone the repository. Then we can load the chains from disk:

```julia
filenames = [
    relpath(outdir(s)) for s in readdir(outdir())
    if occursin(savename(parameters), s) && occursin("seed", s)
]
length(filenames)
```

```julia
chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # Read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # Concatenate them
chains_posterior = chains_posterior[1:3:end] # <= Thin so we're left with 1000 samples
```

```julia
plot(chains_posterior[[:κ, :ϕ, :τ]]; α = .5, linewidth=1.5)
```

```julia
# Compute generated quantities for the chains pooled together
pooled_chains = MCMCChains.pool_chain(chains_posterior)
generated_posterior = vectup2tupvec(generated_quantities(m, pooled_chains));

daily_cases_posterior, daily_deaths_posterior, Rt_posterior, Rt_adj_posterior = generated_posterior;

# Convert `Vector{<:Vector{<:Vector}}` into `Array{<:Real, 3}` with shape `(num_countries, num_days, num_samples)`
daily_cases_posterior_arr = permutedims(arrarrarr2arr(daily_cases_posterior), (2, 3, 1));
daily_deaths_posterior_arr = permutedims(arrarrarr2arr(daily_deaths_posterior), (2, 3, 1));
Rt_posterior_arr = permutedims(arrarrarr2arr(Rt_posterior), (2, 3, 1));
Rt_adj_posterior_arr = permutedims(arrarrarr2arr(Rt_adj_posterior), (2, 3, 1));
```

The posterior predictive distribution:

```julia
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_posterior; main_title = "(posterior)")
```

and with the adjusted \(R_t\):

```julia
ImperialReport13.country_prediction_plot(data, uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior; main_title = "(posterior)")
```

Plotting all countries in one plot can quite cluttered, we're going to use the [`PlotlyJS.jl`](https://github.com/sglyon/PlotlyJS.jl) plotting backend (which generates [plotly.js](https://github.com/plotly/plotly.js/) plots under the hood) which allows us to interactively select which countries to view by clicking on the labels.

```julia
plotlyjs();
```

```julia
ImperialReport13.countries_prediction_plot(data, Rt_posterior_arr; size = (800, 300))
title!("Rt (95% intervals)")
```

```julia
ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_posterior_arr; dims = 2); size = (800, 300))
title!("Expected cases (95% intervals)")
```

```julia
ImperialReport13.countries_prediction_plot(data, cumsum(daily_deaths_posterior_arr; dims = 2); size = (800, 300))
title!("Expected deaths (95% intervals)")
```

```julia
ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_posterior_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected cases / population (95% intervals)")
```

```julia
ImperialReport13.countries_prediction_plot(data, cumsum(daily_deaths_posterior_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected deaths / population (95% intervals)")
```

In the following sections we’re going to back to `PyPlot.jl` as backend (which uses Python’s matplotlib under the hood) instead of `PlotlyJS.jl` since too many `plotly.js` plots can slow down even the finest of computers.

```julia
pyplot()
```


## All countries: prior vs. posterior predictive

For the sake of completeness, here are the prior and posterior predictive distributions for all the 14 countries in a side-by-side comparison.


## What if we didn't do any/certain interventions?

One interesting thing one can do after obtaining estimates for the effect of each of the interventions is to run the model but now *without* all or a subset of the interventions performed. Thus allowing us to get a sense of what the outcome would have been without those interventions, and also whether or not the interventions have the wanted effect.

`turing_data.covariates[m]` is a binary matrix for each `m` (country index), with `turing_data.covariate[m][:, k]` then being a binary vector representing the time-series for the k-th covariate: `0` means the intervention has is not implemented, `1` means that the intervention is implemented. As an example, if schools and universites were closed after the 45th day for country `m`, then `turing_data.covariate[m][1:45, k]` are all zeros and `turing_data.covariate[m][45:end, k]` are all ones.

```julia
# Get the index of schools and univerities closing
schools_universities_closed_index = findfirst(==("schools_universities"), covariate_names)
# Time-series for UK
turing_data.covariates[uk_index][:, schools_universities_closed_index]
```

Notice that the above assumes that not only are schools and universities closed *at some point*, but rather that they also stay closed in the future (at the least the future that we are considering).

Therefore we can for example simulate "what happens if we never closed schools and universities?" by instead setting this entire vector to `0` and re-run the model on the infererred parameters, similar to what we did before to compute the "generated quantities", e.g. \(R_t\).

<details><summary>Convenience function for zeroing out subsets of the interventions</summary>

```julia
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
```

</details>

Now we can consider simulation under the posterior with *no* intervention, and we're going to visualize the respective *portions* of the population by rescaling by total population:

```julia
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
```

```julia
plotlyjs()
ImperialReport13.countries_prediction_plot(data, cumsum(daily_cases_counterfactual_arr; dims = 2); normalize_pop = true, size = (800, 300))
title!("Expected cases / population when no intervention (95% intervals)")
```

Recalling the same figure for the posterior which includes interventions

We can also consider the cases where we only do *some* of the interventions, e.g. we never do a full lockdown (`lockdown`) or close schools and universities (`schools_universities`):

```julia
pyplot()
```

```julia
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
```

As mentioned, this assumes that we will stay in lockdown and schools and universities will be closed in the future. We can also consider, say, removing the lockdown, i.e. opening up, at some future point in time:

```julia
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
```


# Conclusion

Well, there isn't one. As stated before, drawing conclusions is not the purpose of this document. With that being said, we *are* working on exploring this and other models further, e.g. relaxing certain assumptions, model validation & comparison, but this will hopefully be available in a more technical and formal report sometime in the near future after proper validation and analysis. But since time is of the essence in these situations, we thought it was important to make the above and related code available to the public immediately. At the very least it should be comforting to know that two different PPLs both produce the same inference results when the model might be used to inform policy decisions on a national level.

If you have any questions or comments, feel free to reach out either on the [Github repo](https://github.com/TuringLang/Covid19) or to any of us personally.

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> The issue with *not* having the `max` is that it's possible to obtain a negative \(R_t\) which is bad for two reasons: 1) it doesn't make sense with negative spreading of the virus, 2) it leads to invalid parameterization for `NegativeBinomial2`. In Stan a invalid parameterization will be considered a rejected sample, and thus these samples will be rejected. In the case of `Turing.jl`, if \(R_t = 0\), then observing a *positive* number for daily deaths will result in `-Inf` added to the log-pdf and so the sample will also be rejected. Hence, both PPLs will arrive at "correct" inference but with different processes.

<sup><a id="fn.2" class="footnum" href="#fnr.2">2</a></sup> You *could* use something like [Atomic in Julia](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.Atomic), but it comes at a unnecessary performance overhead in this case.

<sup><a id="fn.3" class="footnum" href="#fnr.3">3</a></sup> Recently one of our team-members joined as a maintainer of [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) to make sure that `Turing.jl` also has fast and reliable reverse-mode differentiation. ReverseDiff.jl is already compatible with `Turing.jl`, but we hope that this will help make if much, much faster.