using DrWatson
quickactivate(@__DIR__)

using Pkg
Pkg.instantiate()

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

# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19

# Some other packages we'll need
using Random, Dates, Turing, Bijectors

using Base.Threads
nthreads()

using Pkg
Pkg.status()

data = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"))

countries = data.countries
num_countries = length(countries)
covariate_names = data.covariate_names
lockdown_index = findfirst(==("lockdown"), covariate_names)

model_def = ImperialReport13.model_v2;

parameters = (
    warmup = parsed_args["num-warmup"],
    steps = parsed_args["num-samples"],
    model = "new",
    seed = parsed_args["seed"],
)
Random.seed!(parameters.seed);

# STUFF
turing_data = data.turing_data

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
    lockdown_index
);

@info parameters
chain = sample(m, NUTS(parameters.warmup, 0.95; max_depth=10), parameters.steps + parameters.warmup);

@info "Saving at: $(projectdir("out", savename("chains", parameters, "jls")))"
safesave(projectdir("out", savename("chains", parameters, "jls")), chain)
