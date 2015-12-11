using DataFrames, Murmur3

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
                tail_length = trailing_zeros(Murmur3.x86.hash32(line, seeds[k]))
                max_tail_lengths[k] = max(max_tail_lengths[k], tail_length)
            end
        end
    end

    max_tail_lengths = Φ * 2.^max_tail_lengths

    # take medians of groups and then the mean over them
    group_size = round(Int, num_hashes * group_size_ratio, RoundDown)
    num_groups = round(Int, num_hashes / group_size, RoundUp)

    print(string("Using ", num_hashes, " hashes in ", num_groups,
        " median groups of size ", group_size))
    if num_hashes-group_size*(num_groups-1) < group_size
        print(string(" (last group has size ", num_hashes-group_size*(num_groups-1),
        " only due to rounding)"))
    end

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

iterations = 1
label = "fma"
hash_label = "murmur3"
stream_size = 894069091
num_hashes = 1000
ratio = 0.2

timed_result = @timed fmaF0("data/noblanks.txt", num_hashes, ratio)

result = DataFrame(
    Label = label,
    HashLabel = hash_label,
    InputSize = stream_size,
    Result = timed_result[1],
    Time = timed_result[2],
    Space = timed_result[3],
    Hashes = num_hashes,
    Ratio = ratio)

file_name = string("results/", "results_", label, "_", hash_label, "_k",
    iterations, "_", num_hashes, "_", ratio, ".csv")

writetable(file_name, result)
