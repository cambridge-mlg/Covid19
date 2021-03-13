function setup_data(
    ::typeof(model_v2_zygote),
    turing_data;
    iscuda = false, T = Float64, num_repeat_countries=1
)
    f = iscuda ? cu : identity

    πs = T.(f(repeat(Array(reduce(hcat, turing_data.π)'), outer=(num_repeat_countries, 1))))

    covariates_arr = reduce(turing_data.covariates) do res, c
        cat(res, c; dims = 3)
    end
    covariates_arr = permutedims(covariates_arr, (1, 3, 2));
    covariates_arr = T.(f(repeat(covariates_arr, outer=(1, num_repeat_countries, 1))))

    # Some methods will fail with Int on GPU, so we convert
    cases = map(repeat(turing_data.cases, outer=num_repeat_countries)) do x
        f(T.(x))
    end
    deaths = map(repeat(turing_data.deaths, outer=num_repeat_countries)) do x
        f(T.(x))
    end

    # Order is preserved in `NamedTuple`, so we can do `args...` when using this return value
    # as long as the order is the same as the arguments for the model.
    return (
        num_impute = turing_data.num_impute,
        num_total_days = turing_data.num_total_days,
        cases = cases,
        deaths = deaths,
        π = πs,
        covariates = covariates_arr,
        empidemic_start = repeat(turing_data.epidemic_start, outer=num_repeat_countries),
        population = T.(f(repeat(turing_data.population, outer=num_repeat_countries))),
        serial_intervals = T.(f(turing_data.serial_intervals))
    )
end

function repeat_args(τ, κ, ϕ, y, μ₀, α_hier, ifr_noise; num_repeat_countries = 1)
    return (
        τ = τ, κ = κ, ϕ = ϕ,
        y = repeat(y, outer=num_repeat_countries),
        μ₀ = repeat(μ₀, outer=num_repeat_countries),
        α_hier = α_hier,
        ifr_noise = repeat(ifr_noise, outer=num_repeat_countries)
    )
end

function repeat_args(nt::NamedTuple; num_repeat_countries = 1)
    nt_new = (
        y = repeat(nt.y, outer=num_repeat_countries),
        μ₀ = repeat(nt.μ₀, outer=num_repeat_countries),
        ifr_noise = repeat(nt.ifr_noise, outer=num_repeat_countries)
    )
    return merge(nt, nt_new)
end
