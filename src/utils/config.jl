# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export configdir, read_config, write_config, parse_input_args

# Define configdir.
configdir(args...) = projectdir("config", args...)
!isdir(configdir()) && mkdir(configdir())

# Wite input variables and values into a json file.
function write_config(parsed_args; filename = nothing)
    filename == nothing && (filename = parsed_args["sim_name"] * ".json")
    open(configdir(filename), "w") do f
        write(f, json(parsed_args, 4))
    end
end

# Read input variables and values from a json file.
function read_config(filename)
    configs = JSON.parsefile(configdir(filename); dicttype = Dict)
    for (key, value) in configs
        if typeof(value) == Float64
            configs[key] = convert(Float32, value)
        end
    end
    return configs
end

# Use variables in args to create command line input parser.
function parse_input_args(args)
    s = ArgParseSettings()
    for (key, value) in args
        if typeof(value) == Float64
            add_arg_table(
                s,
                ["--" * key],
                Dict(:default => convert(Float32, value), :arg_type => Float32),
            )
        else
            add_arg_table(
                s,
                ["--" * key],
                Dict(:default => value, :arg_type => typeof(value)),
            )
        end
    end
    return parse_args(s)
end
