module Covid19

using DrWatson, Turing, UnPack
import CUDAExtensions.SpecialFunctions

export ImperialReport13,
    CUDAExtensions,
    NegativeBinomial2,
    GammaMeanCv,
    generated_quantities,
    vectup2tupvec,
    arrarrarr2arr,
    plot_confidence_timeseries,
    plot_confidence_timeseries!

"""
    make_logdensity(model_def, args...)

Makes a method which computes the log density for the given model.

## Arguments
- `model_def`: a model *definition*, not an instance of a model.
- `args`: args usually passed to the model definition to create the model instance.
"""
make_logdensity(model_def, args...) = Turing.Variational.make_logjoint(model_def(args...))

include("io.jl")
include("utils.jl")           # <= stuff that might also be included by sub-modules
include("visualization.jl")   # <= visualization stuff
include("distributions.jl")

# Different related reports
include("imperial-report13/ImperialReport13.jl")

end # module
