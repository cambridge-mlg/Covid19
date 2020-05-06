using Plots, StatsPlots, LaTeXStrings, Dates

# Need some of the impls from this
include("../visualization.jl")


# Ehh, this can be made nicer...
function country_prediction_plot(data::ImperialReport13.Data, country_idx, predictions_country::AbstractMatrix, e_deaths_country::AbstractMatrix, Rt_country::AbstractMatrix; normalize_pop::Bool = false, main_title="")
    pop = data.turing_data.population[country_idx]
    num_total_days = data.turing_data.num_total_days
    num_observed_days = length(data.turing_data.cases[country_idx])

    country_name = data.countries[country_idx]
    start_date = first(data.country_to_dates[country_name])
    dates = cumsum(fill(Day(1), data.turing_data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = data.turing_data.deaths[country_idx]
    daily_cases = data.turing_data.cases[country_idx]
    
    p1 = plot(; xaxis = false, legend = :topleft)
    bar!(preproc(daily_deaths), label="Observed daily deaths")
    title!(replace(country_name, "_" => " ") * " " * main_title)
    vline!([data.turing_data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([num_observed_days], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_country); label = "Expected daily deaths")
    bar!(preproc(daily_deaths), label="Recorded daily deaths (observed)", alpha=0.5)

    p3 = plot(; legend = :bottomleft, xaxis=false)
    plot_confidence_timeseries!(p3, Rt_country; no_label = true)
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.turing_data.covariates[country_idx])))
        if c_time !== nothing
            c_name = data.covariate_names[c_idx]
            if (c_name != "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!(L"$R_t$")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rt_country)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    p4 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_country); label = "Expected daily cases")
    bar!(preproc(daily_cases), label="Recorded daily cases (observed)", alpha=0.5)

    vals = preproc(cumsum(e_deaths_country; dims = 1))
    p5 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p5, vals; label = "Expected deaths")
    plot!(preproc(cumsum(daily_deaths)), label="Recorded deaths (observed)", color=:red)

    vals = preproc(cumsum(predictions_country; dims = 1))
    p6 = plot(; legend = :topleft)
    plot_confidence_timeseries!(p6, vals; label = "Expected cases")
    plot!(preproc(daily_cases), label="Recorded cases (observed)", color=:red)

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200), sharex=true)
    xticks!(1:3:num_total_days, date_strings[1:3:end], xrotation=45)

    return p
end
                                        
function country_prediction_plot(data::ImperialReport13.Data, country_idx, cases, e_deaths, Rt; kwargs...)
    num_observed_days = length(e_deaths)
    e_deaths_country = hcat([e_deaths[t][country_idx] for t = 1:num_observed_days]...)
    Rt_country = hcat([Rt[t][country_idx] for t = 1:num_observed_days]...)
    predictions_country = hcat([cases[t][country_idx] for t = 1:num_observed_days]...)

    return country_prediction_plot(data, country_idx, predictions_country, e_deaths_country, Rt_country; kwargs...)
end

function countries_prediction_plot(data::ImperialReport13.Data, vals::AbstractArray{<:Real, 3}; normalize_pop = false, no_label=false, kwargs...)
    lqs, mqs, hqs = [], [], []
    labels = []

    # `vals` is assumed to be of the shape `(num_countries, num_days, num_samples)`
    num_countries = size(vals, 1)

    for country_idx in 1:num_countries
        val = vals[country_idx, :, :]
        n = size(val, 1)

        pop = data.turing_data.population[country_idx]
        num_total_days = data.turing_data.num_total_days
        num_observed_days = length(data.turing_data.cases[country_idx])

        country_name = data.countries[country_idx]

        # A tiny bit of preprocessing of the data
        preproc(x) = normalize_pop ? x ./ pop : x

        tmp = preproc(val)
        qs = [quantile(tmp[t, :], [0.025, 0.5, 0.975]) for t = 1:n]
        lq, mq, hq = (eachrow(hcat(qs...))..., )

        push!(lqs, lq)
        push!(mqs, mq)
        push!(hqs, hq)
        push!(labels, country_name)
    end

    lqs = reduce(hcat, collect.(lqs))
    mqs = reduce(hcat, collect.(mqs))
    hqs = reduce(hcat, collect.(hqs))

    p = plot(; kwargs...)
    for country_idx in 1:num_countries
        label = no_label ? "" : labels[country_idx]
        plot!(mqs[:, country_idx]; ribbon=(mqs[:, country_idx] - lqs[:, country_idx], hqs[:, country_idx] - mqs[:, country_idx]), label=label)
    end

    return p
end
