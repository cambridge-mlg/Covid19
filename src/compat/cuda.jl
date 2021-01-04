module CUDAExtensions

import CUDA
import SpecialFunctions

# https://github.com/JuliaMath/SpecialFunctions.jl/blob/d18ff04178dd37a60cc716a60ab3caf1a6903f43/src/gamma.jl#L20
function digamma_cu(x)
    if x <= 0 # reflection formula
        ψ = -π / CUDA.tan(π * x)
        x = 1 - x
    else
        ψ = zero(x)
    end
    if x < 7
        # shift using recurrence formula
        ν = one(x)
        n = 7 - CUDA.floor(x)
        while ν <= n - 1
            ψ -= inv(x + ν)
            ν += one(x)
        end
        ψ -= inv(x)
        x += n
    end
    t = inv(x)
    ψ += CUDA.log(x) - t / 2
    t *= t # 1/z^2
    # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
    # TODO: Something really weird going on here. Can't use f32; tried
    # [×] `oftype` of coeffs
    # [×] `Float32` of coeffs
    # [×] `...f0` of coeffs
    # [×] Even failed if I just straight up don't even do this computation o.O
    ψ -= (
        t *
        # @evalpoly(
        #     t,
        #     oftype(x, 0.08333333333333333), oftype(x, -0.008333333333333333),
        #     oftype(x, 0.003968253968253968), oftype(x, -0.004166666666666667),
        #     oftype(x, 0.007575757575757576), oftype(x, -0.021092796092796094),
        #     oftype(x, 0.08333333333333333), oftype(x, -0.4432598039215686)
        # )
        @evalpoly(
            t,
            0.08333333333333333, -0.008333333333333333,
            0.003968253968253968, -0.004166666666666667,
            0.007575757575757576, -0.021092796092796094,
            0.08333333333333333, -0.4432598039215686
        )
    )
    return ψ
end


# Register for replacement in CUDA.jl's broadcasting style
CUDA.cufunc(::typeof(SpecialFunctions.lbeta)) = CUDA.lbeta
CUDA.cufunc(::typeof(SpecialFunctions.lgamma)) = CUDA.lgamma
CUDA.cufunc(::typeof(SpecialFunctions.loggamma)) = CUDA.lgamma
CUDA.cufunc(::typeof(SpecialFunctions.digamma)) = digamma_cu
CUDA.cufunc(::typeof(SpecialFunctions.trigamma)) = CUDA.trigamma

# Includes
include("cuda/distributions.jl")

end
