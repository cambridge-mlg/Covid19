module ImperialReport13

using Turing, Distributions, StatsBase, ArgCheck, UnPack
using TensorOperations
using CUDA

using ..Covid19: NegativeBinomial2, NegativeBinomialVectorized2
import ..Covid19: make_logdensity

include("evolve.jl")
include("models.jl")
include("data.jl")
include("visualization.jl")
include("chainrules.jl")
include("utils.jl")

const model = model_v2 # <= defines the "official" model for this sub-module

end
