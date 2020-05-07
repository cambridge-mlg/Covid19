module ImperialReport13
using Turing, Distributions, StatsBase, ArgCheck

include("../utils.jl")
include("models.jl")
include("data.jl")
include("visualization.jl")

const model = model_v2 # <= defines the "official" model for this sub-module

end
