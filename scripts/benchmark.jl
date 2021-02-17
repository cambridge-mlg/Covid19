using ArgParse

argtable = ArgParseSettings(
    description="Runs benchmarks for CPU and GPU on the specified model."
)
@add_arg_table! argtable begin
    "--samples"
    help = "maximum number of samples to use for benchmarking"
    arg_type = Int
    default = 1_000
    "--seconds"
    help = "maximum seconds to use for benchmarking"
    arg_type = Int
    default = 1
    "--start"
    help = "lowest multiple of countries to benchmark"
    arg_type = Int
    default = 1
    "--end"
    help = "highest multiple of countries to benchmark"
    arg_type = Int
    default = 5
    "--gpu"
    action = :store_true
    "--type"
    default="Float64"
    "--num-blas-threads"
    arg_type = Int
    default = 0
    "--gc"
    action = :store_true
    "model"
    help = "model to use"
    required = true
end

parsed_args = parse_args(ARGS, argtable)
@info parsed_args

using LinearAlgebra
(parsed_args["num-blas-threads"] > 0) && BLAS.set_num_threads(parsed_args["num-blas-threads"])

############
### CODE ###
############
using Covid19
using DrWatson, Turing, Random, CUDA, Zygote
using BenchmarkTools
using ProgressMeter

CUDA.allowscalar(false)

# model_def = ImperialReport13.model_v2_zygote
model_def = eval(Meta.parse(parsed_args["model"]))
@info "Benchmarking $(model_def)" 


function benchmark(model_def, nt, data, num_repeat_countries, T, iscuda; seconds=10, samples=1000, forcegc=false)
    # Setup
    setup_args = ImperialReport13.setup_data(
        model_def,
        data.turing_data;
        T = T, 
        iscuda = iscuda,
        num_repeat_countries = num_repeat_countries
    );
    logπ = ImperialReport13.make_logdensity(model_def, setup_args...)
    args = ImperialReport13.repeat_args(
        nt.τ, nt.κ, nt.ϕ, nt.y, nt.μ₀, nt.α_hier, nt.ifr_noise; 
        num_repeat_countries = num_repeat_countries
    );

    # HACK: attempt at fixing high GC usage causing discepancies.
    # Is this the right thing to do?
    forcegc && GC.gc(true)

    # Benchmark
    eval_results = if iscuda
        @benchmark CUDA.@sync($(logπ)($(args)...)) seconds=seconds samples=samples
    else
        @benchmark $(logπ)($(args)...) seconds=seconds samples=samples
    end

    forcegc && GC.gc(true)

    grad_results = if iscuda
        @benchmark CUDA.@sync($(Zygote.gradient)($(logπ), $(args)...)) seconds=seconds samples=samples
    else
        @benchmark $(Zygote.gradient)($(logπ), $(args)...) seconds=seconds samples=samples
    end
    
    return (eval = eval_results, grad = grad_results)
end

# function benchmark_add!(suite, model_def, nt, data, num_repeat_countries, T, iscuda; seconds=10, samples=1000)
#     # Setup
#     setup_args = ImperialReport13.setup_data(
#         model_def,
#         data.turing_data;
#         T = T, 
#         iscuda = iscuda,
#         num_repeat_countries = num_repeat_countries
#     );
#     logπ = ImperialReport13.make_logdensity(model_def, setup_args...)
#     args = ImperialReport13.repeat_args(
#         nt.τ, nt.κ, nt.ϕ, nt.y, nt.μ₀, nt.α_hier, nt.ifr_noise; 
#         num_repeat_countries = num_repeat_countries
#     );

#     # Benchmark
#     suite["eval"][(T, iscuda)] = @benchmarkable $(logπ)($(args)...)
#     suite["grad"][(T, iscuda)] = @benchmarkable $(Zygote.gradient)($(logπ), $(args)...)
    
#     return suite
# end

##############
### SCRIPT ###
##############
data = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"));

# Instantiate the model so we can get some examples of the latent variables
setup_args = ImperialReport13.setup_data(model_def, data.turing_data);
m = model_def(setup_args...);
vi_cpu = DynamicPPL.VarInfo(m);

# Convert into named tuple for convenience
nt = map(DynamicPPL.tonamedtuple(vi_cpu)) do (v, ks)
    if length(v) == 1
        return first(v)
    else
        return v
    end
end

nt = map(nt) do x
    Float32.(x)
end


# Create one for CUDA
nt_cu = cu(nt)

# Try out
# setup_args = ImperialReport13.setup_data(model_def, data.turing_data; T = Float32, iscuda = true);
# logπ = ImperialReport13.make_logdensity(model_def, setup_args...);

# args = let nt = nt_cu, num_repeat_countries = 1
#     ImperialReport13.repeat_args(
#         nt.τ, nt.κ, nt.ϕ, nt.y, nt.μ₀, nt.α_hier, nt.ifr_noise; 
#         num_repeat_countries = num_repeat_countries
#     )
# end

repeats = map(x -> x^2, parsed_args["start"]:parsed_args["end"])
# repeats = map(x -> x^2, 1:2)
experiments = Dict(:args => parsed_args, :repeats => repeats, :results => Dict())

T = eval(Meta.parse(parsed_args["type"]))
iscuda = parsed_args["gpu"]
nt = iscuda ? nt_cu : nt

experiments[:results] = NamedTuple[]

@showprogress for num_repeat_countries in repeats
    parsed_args["gc"] && GC.gc(true)

    bres = benchmark(
        model_def, nt, data, num_repeat_countries, T, iscuda;
        seconds=parsed_args["seconds"], samples=parsed_args["samples"],
        forcegc=parsed_args["gc"]
    )
    push!(experiments[:results], bres)
end

println(experiments)

# Save
import Dates, JSON
outdir() = projectdir("out")
outdir(args...) = projectdir("out", args...)

basename = "$(Dates.now())_$(savename(parsed_args))"
@info "Saving results in $(basename)"
write(outdir("benchmarks", "$(basename).json"), JSON.json(experiments))
