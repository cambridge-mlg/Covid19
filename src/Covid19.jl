module Covid19

export ImperialReport13,
    NegativeBinomial2,
    GammaMeanCv,
    generated_quantities,
    vectup2tupvec,
    arrarrarr2arr,
    plot_confidence_timeseries,
    plot_confidence_timeseries!

include("utils.jl")           # <= stuff that might also be included by sub-modules
include("utils_overloads.jl") # <= overloads of methods which we DON'T want to include in sub-modules so we don't overload multiple times
include("visualization.jl")   # <= visualization stuff

# Different related reports
include("imperial-report13/ImperialReport13.jl")

end # module
