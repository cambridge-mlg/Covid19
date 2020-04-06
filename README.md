# Covid19.jl

This project is mainly focused on reproducing COVID-19 related models from papers.

## Models
### Imperial College COVID-19 Response Team (Report 13): Estimating the number of infections and the impact of non-pharmaceutical interventions on COVID-19 in 11 European countries
To run:
```julia
# Get the model
using Covid19 # or `include("src/models/imperial_report13.jl")`

# Load the data; results in a variable `data` being available
include("scripts/imperial_report13_load_data.jl")

m = imperial_model_report13(
    data.num_countries,
    data.num_impute,
    data.num_obs_countries,
    data.num_total_days,
    Vector{Vector{Union{Missing, Int64}}}(data.cases),
    Vector{Vector{Union{Missing, Int64}}}(data.deaths),
    data.π,
    data.covariates,
    data.epidemic_start,
    data.serial_intervals
)

chain = sample(m, NUTS(0.65), 1_00) # <= takes a while
```

1. `include("scripts/imperial_report13_load_data.jl")`
2. `using Covid19` or `include("src/models/imperial_report13.jl")`
