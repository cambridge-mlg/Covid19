module Covid19

export ImperialReport13,
    NegativeBinomial2,
    GammaMeanCv

include("utils.jl")

# Different related reports
include("imperial-report13/ImperialReport13.jl")

end # module
