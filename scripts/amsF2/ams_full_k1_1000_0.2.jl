using DataFrames

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
    group_size = round(Int, num_variables * group_size_ratio, RoundDown)
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

iterations = 1
label = "ams"
stream_size = 894069091
num_variables = 1000
ratio = 0.2

timed_result = @timed amsF2("data/noblanks.txt", stream_size,
    num_variables, ratio)

result = DataFrame(
    Label = label,
    InputSize = stream_size,
    Result = timed_result[1],
    Time = timed_result[2],
    Space = timed_result[3],
    Variables = num_variables)

file_name = string("results/", "results_", label, "_k",
    iterations, "_", num_variables, "_", ratio, ".csv")

writetable(file_name, result)
