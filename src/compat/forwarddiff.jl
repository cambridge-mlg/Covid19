#################
# CUDA.jl rules #
#################
# No need to register new adjoints or anything for reverse-mode AD since the use of
# the functions in the adjoints will be correctly replaced in the broadcasting mechanism.

ForwardDiff.DiffRules.@define_diffrule CUDA.lgamma(a) = :(CUDAExtensions.digamma_cu($a))
eval(ForwardDiff.unary_dual_definition(:CUDA, :lgamma))

ForwardDiff.DiffRules.@define_diffrule CUDA.digamma(a) = :(CUDA.trigamma($a))
eval(ForwardDiff.unary_dual_definition(:CUDA, :digamma))

ForwardDiff.DiffRules.@define_diffrule CUDA.lbeta(a, b) = :(CUDA.digamma($a) - CUDA.digamma($a + $b)), :(CUDA.digamma($b) - CUDA.digamma($a + $b))
eval(ForwardDiff.binary_dual_definition(:CUDA, :lbeta))
