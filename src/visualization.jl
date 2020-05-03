using Statistics, Plots, StatsPlots

"""
    plot_confidence_timeseries!(p::Plot, data::AbstractMatrix{<:Real}; label="", kwargs...)

Plots confidence intervals for the time-series represented by `data`. 
Assumes each row corresponds to the samples for a single time-step.
"""
function plot_confidence_timeseries!(p::Plots.Plot, data::AbstractMatrix{<:Real}; no_label = false, label="", kwargs...)
    intervals = [0.025, 0.25, 0.5, 0.75, 0.975]

    qs = [quantile(v, intervals) for v in eachrow(data)]
    llq, lq, mq, uq, uuq = (eachrow(hcat(qs...))..., )
    plot!(mq, ribbon=(mq - llq, uuq - mq), linewidth=0, label=(no_label ? "" : "$(label) (95% quantiles)"), kwargs...)
    plot!(mq, ribbon=(mq - lq, uq - mq), linewidth=0, label=(no_label ? "" : "$(label) (50% quantiles)"), kwargs...)

    return p
end

"""
    plot_confidence_timeseries(data::AbstractMatrix{<:Real}; label="", kwargs...)

See `plot_confidence_timeseries!`.
"""
plot_confidence_timeseries(data::AbstractMatrix{<:Real}; kwargs...) = plot_confidence_timeseries!(plot(), data; kwargs...)
