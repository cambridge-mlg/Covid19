using RData, ArgCheck

struct Data
    stan_data
    turing_data
    country_to_dates
    reported_cases
    countries
    covariate_names
end

function _fix_stan_data_types!(d)
    # Convert some misparsed fields
    d["N2"] = Int(d["N2"]);
    d["N0"] = Int(d["N0"]);

    d["EpidemicStart"] = Int.(d["EpidemicStart"]);

    d["cases"] = Int.(d["cases"]);
    d["deaths"] = Int.(d["deaths"]);

    # Stan will fail if these are `nothing` so we make them empty arrays
    d["x"] = []
    d["features"] = []

    # Add some type-information to arrays (as `-1` is supposed to represent, well, missing data)
    d["deaths"] = Int.(d["deaths"])
    d["cases"] = Int.(d["cases"])

    d["P"] = Int(d["P"]) # `num_covariates`
    d["M"] = Int(d["M"]) # `num_countries`
    d["N0"] = Int(d["N0"]) # `num_impute`
    d["N"] = Int(d["N"]) # `num_obs_countries`
    d["N2"] = Int(d["N2"]) # `num_total_days`

    d["pop"] = Int.(d["pop"]) # `population`

    return d
end

function load_data(path)
    @argcheck endswith(path, ".rds") "$(path) is not a RDS file"
    
    rdata_full = load(path)

    country_to_dates = Dict([(k, rdata_full["dates"][k]) for k in keys(rdata_full["dates"])])
    reported_cases = Dict([(k, Int.(rdata_full["reported_cases"][k])) for k in keys(rdata_full["reported_cases"])])
    
    rdata = rdata_full["stan_data"];

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

    covariate_names = ("schools_universities", "self_isolating_if_ill", "public_events", "any", "lockdown", "social_distancing_encouraged")
    lockdown_index = findfirst(==("lockdown"), covariate_names)

    # `rdata` is a `DictOfVector` so we convert to a simple `Dict` for simplicity
    # NOTE: `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`
    d = Dict([(k, rdata[k]) for k in keys(rdata)])
    _fix_stan_data_types!(d)

    stan_data = copy(d)

    # Rename some columns
    rename!(
        d,
        "f" => "π", "SI" => "serial_intervals", "pop" => "population",
        "M" => "num_countries", "N0" => "num_impute", "N" => "num_obs_countries",
        "N2" => "num_total_days", "EpidemicStart" => "epidemic_start",
        "X" => "covariates", "P" => "num_covariates"
    )

    d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix
    d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix
    d["π"] = colllect(eachcol(d["π"]))  # convert into Array of arrays instead of matrix

    # Can deal with ragged arrays, so we can shave off unobserved data (future) which are just filled with -1
    num_obs_countries = d["num_obs_countries"]
    d["cases"] = collect(d["cases"][1:num_obs_countries[m]] for m = 1:num_countries)
    d["deaths"] = collect(d["deaths"][1:num_obs_countries[m]] for m = 1:num_countries)

    # Convert 3D array into Array{Matrix}
    covariates = [rdata["X"][m, :, :] for m = 1:num_countries]

    turing_data = (; (k => d[String(k)] for k in [:num_countries, :num_impute, :num_obs_countries,
                                           :num_total_days, :cases, :deaths, :π, :epidemic_start,
                                           :population, :serial_intervals])...)
    turing_data = merge(turing_data, (covariates = covariates, ));

    return Data(stan_data, turing_data, country_to_dates, reported_cases, countries, covariate_names)
end
