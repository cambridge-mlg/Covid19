module ImperialReport13
using Turing, Distributions, StatsBase, ArgCheck
using TensorOperations
using CUDA

using ..Covid19: NegativeBinomial2

include("evolve.jl")
include("models.jl")
include("data.jl")
include("visualization.jl")
include("chainrules.jl")

const model = model_v2 # <= defines the "official" model for this sub-module

end
