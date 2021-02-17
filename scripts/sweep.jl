using ArgParse

argtable = ArgParseSettings(
    description="Runs benchmarks for CPU and GPU on the specified model."
)

@add_arg_table! argtable begin
    "--seconds"
    default = 10
    arg_type = Int
    "--start"
    default = 1
    arg_type = Int
    "--end"
    default = 10
    arg_type = Int
    "--num-blas-threads"
    default = 6
    arg_type = Int
    "--gc"
    action = :store_true
end

parsed_args = parse_args(ARGS, argtable)
base_kwargs = parsed_args

using DrWatson
@quickactivate

script = scriptsdir("benchmark.jl")

for (T, usegpu) in [(Float32, false), (Float64, false), (Float32, true)]
    # Add the type and whether or not to use GPU
    d = merge(base_kwargs, Dict("type" => T))
    
    kwargs_strs = map(collect(d)) do (k, v)
        if v isa Bool
            # In case it's a flag, e.g. `--gc`.
            "--$(k)"
        else
            "--$(k)=$(v)"
        end
    end

    usegpu && push!(kwargs_strs, "--gpu")
    
    cmd = `julia-1.5 --project=$(projectdir()) $(script) $(kwargs_strs) ImperialReport13.model_v2_zygote`
    @info "Running '$cmd'"
    run(cmd)
end
