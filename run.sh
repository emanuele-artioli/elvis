#!/bin/bash

# TODO: instead of every possible combination in order, pick random values from the lists
# TODO: potentially, give priority to values that result in a relatively good vmaf. Initialize a weight 

# Define the combinations
videos=("bear" "Big_Buck_Bunny" "breakdance-flare")
scene_numbers=("1")
widths=("640" "960" "1280")
heights=("640" "960")
# TODO: maybe change this into number of vertical and horizontal blocks? 
# Smaller resolutions probably benefit from smaller blocks and vice versa
square_sizes=("16")
to_remove=("20.0")
alpha=("0.5")
smoothing_factor=("0.0" "0.5" "1.0")
neighbor_length=("24")
ref_stride=("2")
subvideo_length=("48")

# Function to check if a combination should be excluded
should_exclude_combination() {
    local video="$1"
    local scene_number="$2"
    local width="$3"
    local height="$4"
    local square_size="$5"
    local blocks="$6"
    local alpha_val="$7"
    local smooth_factor="$8"
    local neighbor="$9"
    local ref="${10}"
    local subvideo="${11}"

    # Add more exclusion rules as needed
    # if [[ <condition> ]]; then
    #     return 0  # Exclude this combination
    # fi

    return 1  # Do not exclude this combination
}

# Iterate over each element of the arrays to generate combinations
for video in "${videos[@]}"; do
    for scene_number in "${scene_numbers[@]}"; do
        for width in "${widths[@]}"; do
            for height in "${heights[@]}"; do
                for square_size in "${square_sizes[@]}"; do
                    for blocks in "${to_remove[@]}"; do
                        for alpha_val in "${alpha[@]}"; do
                            for smooth_factor in "${smoothing_factor[@]}"; do
                                for neighbor in "${neighbor_length[@]}"; do
                                    for ref in "${ref_stride[@]}"; do
                                        for subvideo in "${subvideo_length[@]}"; do
                                            # Check if this combination should be excluded
                                            if should_exclude_combination "$video" "$scene_number" "$width" "$height" "$square_size" "$blocks" "$alpha_val" "$smooth_factor" "$neighbor" "$ref" "$subvideo"; then
                                                echo "Excluding combination: $video, $scene_number, $width, $height, $square_size, $blocks, $alpha_val, $smooth_factor, $neighbor, $ref, $subvideo"
                                                continue
                                            fi

                                            # Call the orchestrator script with the combination, suppressing its output
                                            echo "Running orchestrator with parameters: $video, $scene_number, $width, $height, $square_size, $blocks, $alpha_val, $smooth_factor, $neighbor, $ref, $subvideo"
                                            ./orchestrator.sh "$video" "$scene_number" "$width" "$height" "$square_size" "$blocks" "$alpha_val" "$smooth_factor" "$neighbor" "$ref" "$subvideo" > /dev/null 2>&1
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
