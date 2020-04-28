using Turing, DrWatson

# Allows us to use `safesave(filename, chain)` to ensure that we do not overwrite any chains
DrWatson._wsave(filename, chain::Chains) = write(filename, chain)

"""
    Chains(d::Dict)

Converts a `Dict` into a `Chains`, assuming the values of `d` is either
- `AbstractVector`: samples for a `Real` variable
- `AbstractMatrix`: samples for a `Vector` variable
- `AbstractArray{<:Any, 3}`: samples for a `Matrix` variable
"""
function MCMCChains.Chains(d::Dict)
    vals = []
    names = []
    
    for (k, v) in pairs(d)
        if v isa AbstractVector
            push!(vals, v)
            push!(names, k)
        elseif v isa AbstractMatrix
            push!(vals, v)
            append!(names, ["$(k)[$(i)]" for i = 1:size(v, 2)]) # assuming second dimension is dimensionality
        elseif v isa AbstractArray{<:Any, 3}
            indices = CartesianIndices(v[1, :, :])
            
            # The ordering is such that when you call `reshape` on the vector, the symbols will correspond with
            # the actual indices in the matrix, e.g. `X[i, j]` will be the same as `reshape(X, size)[i, j]`.
            for i = 1:size(indices, 2)
                for j = 1:size(indices, 1)
                    push!(vals, v[:, j, i])
                    push!(names, "$(k)[$j, $i]")
                end
            end
        else
            throw(ArgumentError("I'm so, so sorry but I can't handle $(typeof(v)) :("))
        end
    end

    return Chains(reduce(hcat, vals[2:end]; init = vals[1]), reduce(vcat, names[2:end]; init = names[1]))
end
