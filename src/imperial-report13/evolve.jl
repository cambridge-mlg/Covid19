function evolve(
    y, μ, α,
    serial_intervals, population, covariates,
    num_impute, num_total_days
)
    mod = y isa CuArray ? CUDA : Base

    num_countries = length(population)

    # Pre-allocate `daily_cases_pred` and `cases_pred`
    T = typejoin(eltype(y), eltype(μ), eltype(α)) # TODO: improve?
    daily_cases_pred = mod.zeros(T, num_countries, num_total_days)
    cases_pred = mod.zeros(T, num_countries, num_total_days)

    daily_cases_pred[:, 1:num_impute] .= y
    cases_pred[:, 1:num_impute] = cumsum(daily_cases_pred[:, 1:num_impute]; dims = 2) .- y

    for t = (num_impute + 1):num_total_days
        cases_pred[:, t] = cases_pred[:, t - 1] + daily_cases_pred[:, t - 1]

        Rₜ = μ .* exp.(covariates[t, :, :] * (-α))
        Rₜ_adj = (clamp.(population .- cases_pred[:, t], zero(T), Inf) ./ population) .* Rₜ

        ts_prev = 1:t - 1
        r = daily_cases_pred[:, ts_prev] * serial_intervals[reverse(ts_prev)]
        daily_cases_pred[:, t] = Rₜ_adj .* r
    end

    return daily_cases_pred
end


function ∂evolve(y, μ, α, serial_intervals, population, covariates, num_impute, num_total_days)
    mod = y isa CuArray ? CUDA : Base

    num_countries = length(population)
    num_covariates = length(α)

    # Pre-allocate `daily_cases_pred` and `cases_pred`
    T = typejoin(eltype(y), eltype(μ), eltype(α)) # TODO: improve?
    daily_cases_pred = mod.zeros(T, num_countries, num_total_days)
    # cases_pred = mod.zeros(T, num_countries, num_total_days)

    # Pre-allocate the gradients
    ∂daily_cases_pred = (
        # For first `num_impute` time-steps
        # ∂c∂y = 1 since first `num_impute` are filled with `y`
        # ∂c∂μ = 0 since first `num_impute` are filled with `y`
        y = mod.ones(T, num_countries, num_total_days),
        μ = mod.zeros(T, num_countries, num_total_days),
        α = mod.zeros(T, num_countries, num_total_days, num_covariates)
    )

    # Compute `daily_cases_pred` and `cases_pred`
    daily_cases_pred[:, 1:num_impute] .= y
    # cases_pred[:, 1:num_impute] = cumsum(daily_cases_pred[:, 1:num_impute]; dims = 2) .- y

    for t = (num_impute + 1):num_total_days
        # cases_pred[:, t] = cases_pred[:, t - 1] + daily_cases_pred[:, t - 1]
        # cases_pred_t = cases_pred[:, t]
        cases_pred_t = sum(daily_cases_pred[:, 1:t - 1]; dims = 2)

        Rₜ = μ .* exp.(covariates[t, :, :] * (-α))
        R_scale = (clamp.(population .- cases_pred_t, zero(T), Inf) ./ population)
        Rₜ_adj = R_scale .* Rₜ

        ts_prev = 1:t - 1
        r = daily_cases_pred[:, ts_prev] * serial_intervals[reverse(ts_prev)]
        daily_cases_pred[:, t] = Rₜ_adj .* r

        # ∂c∂y
        ∂R̂∂y = - (Rₜ ./ population) .* sum(∂daily_cases_pred.y[:, ts_prev], dims=2)
        ∂daily_cases_pred.y[:, t] = (
            ∂R̂∂y .* r
            + Rₜ_adj .* (∂daily_cases_pred.y[:, ts_prev] * serial_intervals[reverse(ts_prev)])
        )

        # ∂c∂μ
        ∂R̂∂u = (Rₜ_adj ./ μ) - (Rₜ ./ population) .* sum(∂daily_cases_pred.μ[:, ts_prev], dims=2)
        ∂daily_cases_pred.μ[:, t] = (
            ∂R̂∂u .* r
            + Rₜ_adj .* (∂daily_cases_pred.μ[:, ts_prev] * serial_intervals[reverse(ts_prev)])
        )

        # ∂c∂α
        # TODO: improve?
        ∂R∂α = - Rₜ .* covariates[t, :, :] # (num_countries, num_covariates)
        tmp = dropdims(sum(∂daily_cases_pred.α; dims=2); dims=2)
        ∂R̂∂α = R_scale .* ∂R∂α .- (Rₜ ./ population) .* tmp

        s = let a = ∂daily_cases_pred.α[:, ts_prev, :], b = serial_intervals[reverse(ts_prev)]
            @tensor begin
                s[m, k] := a[m, τ, k] * b[τ]
            end
        end
        ∂α = ∂R̂∂α .* r + Rₜ_adj .* s

        for k = 1:num_covariates
            ∂daily_cases_pred.α[:, t, k] = ∂α[:, k]
        end
    end

    return daily_cases_pred, ∂daily_cases_pred
end
