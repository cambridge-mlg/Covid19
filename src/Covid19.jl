module Covid19

export ImperialReport13,
    NegativeBinomial2,
    GammaMeanCv

# Fix for stuff
using Distributions, Random, DistributionsAD

function Distributions.rand(rng::Random.AbstractRNG, d::DistributionsAD.FillVectorOfUnivariate)
    return rand(rng, d.v.value, length(d))
end

include("utils.jl")

# Different related reports
include("imperial-report13/ImperialReport13.jl")

end # module
