using DataFrames, ProgressMeter

# read input arguments

(num_variables, iterations, stream_size, input_file) = try
  (parse(Int,ARGS[1]), parse(Int,ARGS[2]), parse(Int,ARGS[3]), ARGS[4])
catch
  throw(string("Usage: julia ams_iterate-ratio.jl ",
    "<num_variables> <iterations> <stream_size> <input_file>"))
end

# prepare output

label = "ams"
name = "Alon-Matias-Szegedy"

path = string("results/", label, "_iterate-ratio")
if ispath(path) == false
  mkpath(path)
end
output_file = string(path,"/input", stream_size, "_numvars", num_variables,
  "_it", iterations, ".csv")

# give info

println("Running ", name, " with iterating ratio:")
println(" - num_variables: ", num_variables)
println(" - iterations: ", iterations)
println(" - stream_size: ", stream_size)
println(" - input_file: ", input_file)
println(" > ratio in range 0.1:0.1:0.9")
println(" > output file: ", output_file)

### Alon-Matias-Szegedy Algorithm for F2

function amsF2(file, stream_size::Int, num_variables::Int,
    group_size_ratio::Float64)

    # choose positions of variables in stream uniformly at random
    positions = sample(1:stream_size, num_variables, replace=false)

    # Dict that stores our variables and their values
    dict = Dict()

    open(file) do filehandle
        for (index, value) in enumerate(eachline(filehandle))

            # if the current stream element isn't in our Dict
            if get(dict, value, 0) == 0
                # check if we reached a position in the stream to
                # initialize our next variable
                if in(index, positions)
                    # store the element in our Dict
                    dict[value] = 1
                end
            else
                # the element is in our Dict, increase its count
                dict[value] += 1
            end

        end
    end

    # calculate estimates
    estimates = stream_size .* (collect(values(dict)) .*2 - 1)

    # check if all our variables are unique
    if length(estimates) < num_variables
        # println("Only ", length(estimates), " of ", num_variables,
        #   " variables are unique, using only those for estimation.")
          num_variables = length(estimates)
    end

    # take medians of groups and then the mean over them
    group_size = min(round(Int, num_variables * group_size_ratio, RoundDown), 1)
    num_groups = round(Int, num_variables / group_size, RoundUp)

    medians = zeros(num_groups)
    for i in 1:num_groups
        group_start = (i-1)*group_size + 1
        group_end = i*group_size

        # the last group is smaller:
        if i == num_groups
            group_end = num_variables
        end
        medians[i] = median(estimates[group_start:group_end])

    end
    return mean(medians)
end


### Evaluation

results = DataFrame(
    Label = AbstractString[],
    InputSize = Int[],
    Result = Float64[],
    Time = Float64[],
    Space = Float64[],
    Variables = Int[],
    Ratio = Float64[])

@showprogress for ratio in 0.1:0.1:0.9
  for k in 1:iterations

    try
      timed_result = @timed amsF2(input_file, stream_size,
        round(Int, num_variables),ratio)

        result = DataFrame(
          Label = label,
          InputSize = stream_size,
          Result = timed_result[1],
          Time = timed_result[2],
          Space = timed_result[3],
          Variables = num_variables,
          Ratio = ratio)

        results = vcat(results, result)
    catch e
        println("Could not use ratio ", string(ratio), ": ", e)
    end
  end
end

writetable(output_file, results)
