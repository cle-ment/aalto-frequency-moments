using DataFrames, ProgressMeter

# read input arguments

(num_hashes, iterations, stream_size, input_file) = try
  (parse(Int,ARGS[1]), parse(Int,ARGS[2]), parse(Int,ARGS[3]), ARGS[4])
catch
  throw(string("Usage: julia fma_poly_iterate-ratio.jl ",
    "<num_hashes> <iterations> <stream_size> <input_file>"))
end

# prepare output

label = "fma_poly"
name = "Flajolet-Martin"

path = string("results/", label, "_iterate-ratio")
if !(ispath(path))
  mkpath(path)
end
output_file = string(path,"/input", stream_size, "_numvars", num_hashes,
  "_it", iterations, ".csv")

# give info

println("Running ", name, " with iterating ratio:")
println(" - num_hashes: ", num_hashes)
println(" - iterations: ", iterations)
println(" - stream_size: ", stream_size)
println(" - input_file: ", input_file)
println(" > ratio in range 0.1:0.5:0.95")
println(" > output file: ", output_file)

### Polynomial hash function

function polyhash(string::AbstractString, seed::Int, bits::Int)
    maxval = 2^bits
    hash = maxval
    for char in collect(string)
        hash += Int(char) ^ seed + 42
    end
    return mod(hash, maxval)
end;

### Flajolet-Martin F0 Algorithm

function fmaF0(file, num_hashes::Int, group_size_ratio::Float64)

    if group_size_ratio <= 0 || group_size_ratio > 1
        error("Choose parameter group_size_ratio in interval (0..1]")
        return
    end

    Φ = 0.77351 # correction factor, E(R) = log2Φn

    seeds = rand(1:num_hashes, num_hashes)
    max_tail_lengths = zeros(num_hashes)

    open(file) do filehandle
        for line in eachline(filehandle)
            for k in 1:length(seeds)
                tail_length = trailing_zeros(polyhash(line, seeds[k], 30))
                max_tail_lengths[k] = max(max_tail_lengths[k], tail_length)
            end
        end
    end

    max_tail_lengths = Φ * 2.^max_tail_lengths

    # take medians of groups and then the mean over them
    group_size = round(Int, num_hashes * group_size_ratio, RoundDown)
    num_groups = round(Int, num_hashes / group_size, RoundUp)

    medians = zeros(num_groups)
    for i in 1:num_groups
        group_start = (i-1)*group_size + 1
        group_end = i*group_size

        # the last group is smaller:
        if i == num_groups
            group_end = num_hashes
        end

        medians[i] = median(max_tail_lengths[group_start:group_end])
    end
    return mean(medians)
end;

### Evaluation

results = DataFrame(
    Label = AbstractString[],
    InputSize = Int[],
    Result = Float64[],
    Time = Float64[],
    Space = Float64[],
    Hashes = Int[],
    Ratio = Float64[])

@showprogress for ratio in 0.1:0.05:1
  for k in 1:iterations

    try
      timed_result = @timed fmaF0(input_file, round(Int, num_hashes),ratio)

        result = DataFrame(
          Label = label,
          InputSize = stream_size,
          Result = timed_result[1],
          Time = timed_result[2],
          Space = timed_result[3],
          Hashes = num_hashes,
          Ratio = ratio)

        results = vcat(results, result)
    catch e
        println("Could not use ratio ", string(ratio), ": ", e)
    end
  end
end

writetable(output_file, results)
