using DataFrames

### Tools

toy_data_counts = [10, 100, 1000, 10000, 100000, 1000000]
toy_data_files = [
  "data/noblanks-10.txt",
  "data/noblanks-100.txt",
  "data/noblanks-1000.txt",
  "data/noblanks-10000.txt",
  "data/noblanks-100000.txt",
  "data/noblanks-1000000.txt"]
toy_data_dict = Dict{Int, AbstractString}(zip(toy_data_counts, toy_data_files));

macro measureCall(function_call, expected_result,
    input_size, iterations, label)

    local local_expected_result = eval(expected_result)
    local local_input_size = eval(input_size)
    local local_iterations = eval(iterations)
    local local_label = eval(label)

    local result_mean = 0
    local squared_error_mean = 0
    local time_mean = 0.0
    local space_mean = 0.0

    for i = 1:iterations
        timed_result = @timed(eval(function_call))
        result_mean += timed_result[1]
        squared_error_mean += (local_expected_result - timed_result[1])^2
        time_mean += timed_result[2]
        space_mean += timed_result[3]
    end

    result_mean = result_mean / local_iterations
    squared_error_mean = sqrt(squared_error_mean / local_iterations)
    time_mean = time_mean / local_iterations
    space_mean = space_mean / local_iterations

    return DataFrame(
        Label = local_label,
        InputSize = local_input_size,
        Iterations= local_iterations,
        Result = result_mean,
        Error = squared_error_mean,
        Time = time_mean,
        Space = space_mean)
end

function denormalizeMeasurements(df::DataFrame)
    labels_error = fill("Error",size(df,1))
    labels_time = fill("Time",size(df,1))
    labels_space = fill("Space",size(df,1))

    return rename!(vcat(
        hcat(rename!(df[:,
            [:Label, :InputSize, :Iterations, :Error]],
            :Error, :Measurement), labels_error),
        hcat(rename!(
            df[:, [:Label, :InputSize, :Iterations, :Time]],
            :Time, :Measurement), labels_time),
        hcat(rename!(
            df[:, [:Label, :InputSize, :Iterations, :Space]],
            :Space, :Measurement), labels_space)
        ), :x1, :Type)
end

### One Pass F0 algorithm

function count_words(stream::IOStream, inputSize=0)
    wordcounts = Dict{UTF8String,Int}();
    # preallocate memory
    if inputSize > 0
        sizehint!(wordcounts, inputSize)
    end
    while !eof(stream)
        word = readline(stream)
        wordcounts[word] = get(wordcounts, word, 0) + 1
    end
    return collect(values(wordcounts))
end;

onePassF0(file) = length(open(count_words, file))

### Evaluation

label_onePassF0 = "One Pass F0"

k = 3 # iterations;

results_onePassF0 = DataFrame(
    Label = AbstractString[],
    InputSize = Int[],
    Iterations= Int[],
    Result = Float16[],
    Error = Float16[],
    Time = Float16[],
    Space = Float16[])

for i in 1:length(toy_data_files)
    results_onePassF0 = vcat(results_onePassF0,
    @eval @measureCall(
    onePassF0(toy_data_files[$i]),
    0, # baseline_results_F0[$i],
    10^$i, $k, $label_onePassF0))
end

writetable("results/results_onePassF0-toydata-k3.csv", results_onePassF0)
