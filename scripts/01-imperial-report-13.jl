using DrWatson
quickactivate(@__DIR__)

import DrWatson: datadir

datadir() = projectdir("data", "imperial-report13")
datadir(s...) = projectdir("data", "imperial-report13", s...)

using ArgParse

argtable = ArgParseSettings()
@add_arg_table! argtable begin
    "--chunksize"
        help = "chunksize to be used by ForwardDiff.jl"
        arg_type = Int
        default = 40
    "--num-samples", "-n"
        help = "number of samples"
        arg_type = Int
        default = 3000
    "--num-warmup", "-w"
        help = "number of samples to use for warmup/adaptation"
        arg_type = Int
        default = 1000
    "seed"
        help = "random seed to use"
        required = true
        arg_type = Int
end

parsed_args = parse_args(ARGS, argtable)

# ENV["PYTHON"] = "$(ENV['HOME'])/.local/bin/python"

# using Pkg
# Pkg.build("PyCall")

using Covid19

using RData
using Random, Turing, Bijectors

Turing.setchunksize(parsed_args["chunksize"])

using Base.Threads
@info "Using $(nthreads()) threads"

begin
    dist = product_distribution(fill(truncated(Normal(3.28, 10), 0, Inf), 14))
    b = bijector(dist)
    x = rand(dist)

    # stupid check to make sure that we have the correct versions of Bijectors.jl used WITHIN Turing.jl
    @assert (invlink(dist, b(x)) ≈ Turing.invlink(dist, b(x))) && (Turing.invlink(dist, b(x)) ≈ x)
end

@info "Loading and processing data $(datadir("processed.rds"))"
rdata = load(datadir("processed.rds"));

# Convert some misparsed fields
# rdata["covariate4"] = Matrix(rdata["covariate4"]);

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

# TODO: not sure if either of the below is the correct ordering of the names!
# names_covariates = (
#     "Schools + Universities",
#     "Self-isolating if ill",
#     "Public events",
#     "Lockdown",
#     "Social distancing encouraged"
# )
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

model_def = ImperialReport13.model_v2_vectorized_multithreaded

parameters = (
    warmup = parsed_args["num-warmup"],
    steps = parsed_args["num-samples"],
    model = "imperial-report13-v2-vectorized-non-predict-$(nthreads())-threads",
    seed = parsed_args["seed"],
    with_lockdown = true
)
Random.seed!(parameters.seed);

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
    lockdown_index
);

@info parameters
chain = sample(m, NUTS(parameters.warmup, 0.95; max_depth=10), parameters.steps + parameters.warmup);

@info "Saving at: $(projectdir("out", savename("chains", parameters, "jls")))"
safesave(projectdir("out", savename("chains", parameters, "jls")), chain)
