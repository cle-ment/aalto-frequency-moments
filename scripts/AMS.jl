module AMS

  # Alon-Matias-Szegedy Algorithm for F2
  # as in Alon, Matias, Szegedy: section 2.1: "Estimating F_k"

  function ams(file, stream_size::Int, num_variables::Int,
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
end
