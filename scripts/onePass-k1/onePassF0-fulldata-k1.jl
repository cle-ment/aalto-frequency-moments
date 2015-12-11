using DataFrames

### Tools

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

k = 1 # iterations
i = 894069091

results_onePassF0 = @eval @measureCall(
    onePassF0("data/noblanks.txt"),
    0, # baseline_results_F0[$i],
    $i, $k, $label_onePassF0)

writetable("results/results_onePassF0-fulldata-k1.csv", results_onePassF0)
