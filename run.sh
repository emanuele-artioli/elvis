#!/bin/bash

# Define the combinations
videos=("bear")
scene_numbers=("1")
widths=("432")
heights=("240")
# TODO: maybe change this into number of vertical and horizontal blocks? 
# Smaller resolutions probably benefit from smaller blocks and vice versa
square_sizes=("16")
blocks_to_remove=("0.5")
alpha=("0.5")
neighbor_length=("20" "48")
ref_stride=("6")
subvideo_length=("24")

# Function to check if a combination should be excluded
should_exclude_combination() {
    local video="$1"
    local scene_number="$2"
    local width="$3"
    local height="$4"
    local square_size="$5"
    local blocks="$6"
    local alpha_val="$7"
    local neighbor="$8"
    local ref="$9"
    local subvideo="${10}"

    # Add more exclusion rules as needed
    # if [[ <condition> ]]; then
    #     return 0  # Exclude this combination
    # fi

    if [[ "$width" == "640" && "$height" != "360" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$width" == "960" && "$height" != "540" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$width" == "1280" && "$height" != "720" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$width" == "1920" && "$height" != "1080" ]]; then
        return 0  # Exclude this combination
    fi

    return 1  # Do not exclude this combination
}

# Iterate over each element of the arrays to generate combinations
for video in "${videos[@]}"; do
    for scene_number in "${scene_numbers[@]}"; do
        for width in "${widths[@]}"; do
            for height in "${heights[@]}"; do
                for square_size in "${square_sizes[@]}"; do
                    for blocks in "${blocks_to_remove[@]}"; do
                        for alpha_val in "${alpha[@]}"; do
                            for neighbor in "${neighbor_length[@]}"; do
                                for ref in "${ref_stride[@]}"; do
                                    for subvideo in "${subvideo_length[@]}"; do
                                        # Check if this combination should be excluded
                                        if should_exclude_combination "$video" "$scene_number" "$width" "$height" "$square_size" "$blocks" "$alpha_val" "$neighbor" "$ref" "$subvideo"; then
                                            echo "Excluding combination: $video, $scene_number, $width, $height, $square_size, $blocks, $alpha_val, $neighbor, $ref, $subvideo"
                                            continue
                                        fi

                                        # Call the orchestrator script with the combination, suppressing its output
                                        echo "Running orchestrator with parameters: $video, $scene_number, $width, $height, $square_size, $blocks, $alpha_val, $neighbor, $ref, $subvideo"
                                        ./orchestrator.sh "$video" "$scene_number" "$width" "$height" "$square_size" "$blocks" "$alpha_val" "$neighbor" "$ref" "$subvideo" > /dev/null 2>&1
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
