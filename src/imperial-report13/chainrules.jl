import ChainRulesCore
import FillArrays
import TensorOperations


function ChainRulesCore.rrule(
    ::typeof(evolve),
    y, μ, α,
    serial_intervals, population, covariates,
    num_impute, num_total_days
)
    daily_cases_pred, ∂daily_cases_pred = ∂evolve(
        y, μ, α,
        serial_intervals, population, covariates,
        num_impute, num_total_days
    )

    function evolve_pullback(Δ)
        # HACK: `TensorOperations.jl` does not support `Fill` :/
        ∇α = if Δ isa FillArrays.Fill
            vec(sum(Δ.value * ∂daily_cases_pred.α; dims = [1, 2]))
        else
            @tensor begin
                ∇α[k] := Δ[m, t] * ∂daily_cases_pred.α[m, t, k]
            end
        end

        return (
            NO_FIELDS,
            vec(sum(Δ .* ∂daily_cases_pred.y, dims=2)),
            vec(sum(Δ .* ∂daily_cases_pred.μ, dims=2)),
            ∇α,
            Zero(),
            Zero(),
            Zero(),
            Zero(),
            Zero()
        )
    end

    return daily_cases_pred, evolve_pullback
end
