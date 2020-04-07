#################################
# Loading and transforming data #
#################################
using Dates
using DataFrames, CSV

using DrWatson
using LibGit2

# TODO: make this a much nicer "DataLoader" or something

const IMPERIAL_PROJ_DIR = projectdir("external/covid19model/")
const DATA_DIR = joinpath(IMPERIAL_PROJ_DIR, "data")

if !ispath(DATA_DIR)
    mkpath(projectdir("external"))
    LibGit2.clone("https://github.com/ImperialCollegeLondon/covid19model.git", IMPERIAL_PROJ_DIR)
end

df = CSV.read(joinpath(DATA_DIR, "COVID-19-up-to-date.csv"))

countries = [
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
  "Switzerland"
]

readdir(DATA_DIR)

# Infection-fatality-rate
ifr_by_country = CSV.read(joinpath(DATA_DIR, "weighted_fatality.csv"); copycols=true)
ifr_by_country[:, :country] = map(x -> x == "United Kingdom" ? "United_Kingdom" : x , ifr_by_country[:, :country])

serial_intervals = CSV.read(joinpath(DATA_DIR, "serial_interval.csv"))

# Only the first 11 rows are relevant (look at the CSV-file and you'll see)
covariates = CSV.read(joinpath(DATA_DIR, "interventions.csv"); limit=11, copycols=true)
covariates = covariates[1:11, 1:8]
names(covariates)
covariates.self_isolating_if_ill

## Make all covariates that occurr after lockdown have the same date as the lockdown
for col in filter(!=(:lockdown), names(covariates)[2:end])
    covariates[covariates[!, col] .> covariates.lockdown, col] .= covariates[covariates[!, col] .> covariates.lockdown, :lockdown]
end

# To store stuff
dates = Dict()
reported_cases = Dict()
deaths_by_country = Dict()

num_total_days_default = 75

data = (
    epidemic_start = [],
    num_obs_countries = [],
    cases = [],
    deaths = [],
    covariates = [],
    π = [],
    y = [],
    serial_intervals = copy(serial_intervals.fit[1:num_total_days_default])
)

for country in countries
    num_total_days = num_total_days_default
    # Should only return a single number for each country
    IFR = first(ifr_by_country[ifr_by_country.country .== country, :weighted_fatality])
    covariates1 = covariates[covariates.Country .== country, 2:8]

    # Filter the country and sort by date
    df_country = df[df.countriesAndTerritories .== country, :] # filter by country
    df_country.date = Date.(df_country.dateRep, "d/m/y")              # parse dates
    df_country = sort(df_country, :date)                              # sort by time

    # Compute the "beginning" of the epidemic in `country`
    index_first_case = findfirst(==(1), df_country.cases .> 0)            # find first index where we have cases
    index_first10_deaths = findfirst(cumsum(df_country.deaths) .≥ 10)        # find index where deaths ≥ 10
    index_first10_deaths_month_prior = index_first10_deaths - 30                                # consider the month before reaching 10 deaths

    println("First non-zero cases is on day $(index_first_case), and 30 days before 5 days is day $(index_first10_deaths_month_prior)")
    df_country = df_country[index_first10_deaths_month_prior:end, :]

    # FIXME: What if `index_first10_deaths + 1 - index_first10_deaths_month_prior` < 1? Shouldn't we have a `max` here?
    # TODO: uhmm this is always going to be `31`? Since `index_first10_deaths_month_prior`
    # is defined to be index_first10_deaths - 30 o.O
    push!(data.epidemic_start, index_first10_deaths + 1 - index_first10_deaths_month_prior)

    for covariate in names(covariates1)
        df_country[!, covariate] .= (df_country.date .≥ covariates1[1, covariate])
    end

    dates[country] = df_country.date

    # Hazard estimation
    N = size(df_country, 1)
    println("$country has $N days of data")
    forecast = num_total_days - N

    if forecast < 0
        # TODO: allow "forecasting" for dates with observations so we can compare
        @warn "num_total_days ≤ N, i.e. trying to forecast into the past"
        num_total_days = N
        forecast = 0
    end

    ## Computation the infection-to-death distribution (πₘ)
    mean1, cv1 = 5.1, 0.86 # infection to onset of symptoms
    mean2, cv2 = 18.8, 0.45 # onset of symptoms to death
    
    # Assuming IFR is probability of dying given infection
    x1 = rand(GammaMeanCv(mean1, cv1), Int(5e6))
    x2 = rand(GammaMeanCv(mean2, cv2), Int(5e6))

    empirical_cdf = StatsBase.ecdf(x1 + x2)
    survival_function(u) = IFR * empirical_cdf(u)

    # THE FOLLOWING SECTION HAS BEEN REPLACED BY A MUCH SIMPLER IMPL

    # h = zeros(forecast + N)
    # h[1] = survival_function(1.5) - survival_function(0.)
    # for t = 2:length(h)
    #     # We're scaling the discretization by the "remainder" of `F`
    #     # 
    #     #   (F(i + 0.5) - F(i - 0.5)) / (1 - F(i - 0.5))
    #     # 
    #     h[t] = (survival_function(t + 0.5) - survival_function(t - 0.5)) / (1 - survival_function(t - 0.5))
    # end

    # # [1, 1 * (1 - hᵢ), 1 * (1 - hᵢ) * (1 - h₂), …, ∏ᵢ (1 - hᵢ)]

    # s = zeros(num_total_days)
    # s[1] = 1.
    # for t = 2:num_total_days
    #     s[t] = s[t - 1] * (1 - h[t - 1])
    # end

    # # Consider
    # #    sₜ = sₜ₋₁ * ((1 - Fₘ(t + 0.5)) / (1 - Fₘ(t - 0.5)))
    # #
    # # since
    # #
    # #    1 - hₜ₋₁ = 1 - (Fₘ(t - 0.5) - Fₘ(t - 1.5)) / (1 - Fₘ(t - 1.5))
    # #             = ((1 - Fₘ(t - 1.5)) - (Fₘ(t - 0.5) - Fₘ(t - 1.5))) / (1 - Fₘ(t - 1.5))
    # #             = (1 - Fₘ(t - 0.5)) / (1 - Fₘ(t - 1.5))
    # #
    # # So then
    # #
    # #    s₃ = s₂ * (1 - h₂)
    # #       = (1 - h₁) * (1 - h₂)
    # #       = ((1 - Fₘ(1.5)) / (1 - Fₘ(0))) * ((1 - Fₘ(2.5)) / (1 - Fₘ(1.5)))
    # #       = (1 - Fₘ(2.5)) / (1 - Fₘ(0))
    # #
    # # but Fₘ(0) = 0 soooo it doesn't matter? So then we have
    # #
    # #    sᵢ * hᵢ = (1 - Fₘ(t - 0.5)) * (Fₘ(t + 0.5) - Fₘ(t - 0.5)) / (1 - Fₘ(t - 0.5))
    # #            = (Fₘ(t + 0.5) - Fₘ(t - 0.5))
    # #
    # # ¿Que? πₛₘ = Fₘ(s + 0.5) - Fₘ(s - 0.5)

    # πₘ = s .* h
    
    πₘ = map(1:num_total_days) do i
        if i == 1
            survival_function(1.5) - survival_function(0.5)
        else
            survival_function(i + 0.5) - survival_function(i - 0.5)
        end
    end

    ## If you want to verify:
    # norm(πₘ - f)^2 / (norm(πₘ) * norm(f))

    # Add `missing` for the days we're supposed to forecast
    y = vcat(df_country.cases, fill(missing, forecast))
    deaths = vcat(df_country.deaths, fill(missing, forecast))
    cases = vcat(df_country.cases, fill(missing, forecast))

    reported_cases[country] = df_country.cases
    deaths_by_country[country] = df_country.deaths

    covariates2 = copy(df_country[!, names(covariates1)])
    for i = N + 1:(N + forecast)
        push!(covariates2, covariates2[N, :])
    end

    # Append the data
    push!(data.num_obs_countries, N)
    push!(data.y, y[1])  # TODO: what is this?
    push!(data.covariates, Matrix(covariates2[:, 1:7]))
    push!(data.π, πₘ)
    push!(data.deaths, deaths)
    push!(data.cases, cases)
end

data = merge(
    data,
    (num_countries = length(data.covariates), num_impute = 6, num_total_days = num_total_days_default)
)

using Serialization
mkpath(projectdir("out"))
open(io -> serialize(io, data), projectdir("out", "data.jls"), "w")


# TODO: save the data
# @assert data.num_obs_countries .== [37, 62, 43, 49, 46, 51, 34, 40, 36, 40, 44]
data
