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

### One Pass F2 algorithm

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

square(x) = x^2
@vectorize_1arg Number square;

onePassF2(file) = sum(square(open(count_words, file)));

### Evaluation

label_onePassF2 = "One Pass F2 with blanklines"

k = 3 # iterations;
i = 911740903

results_onePassF2 = @eval @measureCall(
    onePassF2("data/twitter_words.txt"),
    0, # baseline_results_F2[$i],
    $i, $k, $label_onePassF2)

writetable("results/results_onePassF2-fulldata-blanks-k3.csv", results_onePassF2)
