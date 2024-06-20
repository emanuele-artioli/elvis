#!/bin/bash

# TODO: add time to run to metrics file

# Define the combinations
videos=("Tears_of_Steel_4k")
scene_numbers=("9")
widths=("1920")
heights=("1080")
# TODO: maybe change this into number of vertical and horizontal blocks? 
# Smaller resolutions probably benefit from smaller blocks and vice versa
square_sizes=("16" "20" "40")
horizontal_stride=("2")
vertical_stride=("2")
neighbor_length=("2" "3" "5" "10")
ref_stride=("2" "3" "5" "10")
subvideo_length=("5" "10" "15" "25" "50")

# Function to check if a combination should be excluded
should_exclude_combination() {
    local video="$1"
    local scene_number="$2"
    local width="$3"
    local height="$4"
    local square_size="$5"
    local h_stride="$6"
    local v_stride="$7"
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

    # TODO: let's keep it to see if it actually ends up in the same video as original
    if [[ "$h_stride" == "1" && "$v_stride" == "1" ]]; then
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
                    for h_stride in "${horizontal_stride[@]}"; do
                        for v_stride in "${vertical_stride[@]}"; do
                            for neighbor in "${neighbor_length[@]}"; do
                                for ref in "${ref_stride[@]}"; do
                                    for subvideo in "${subvideo_length[@]}"; do
                                        # Check if this combination should be excluded
                                        if should_exclude_combination "$video" "$scene_number" "$width" "$height" "$square_size" "$h_stride" "$v_stride" "$neighbor" "$ref" "$subvideo"; then
                                            echo "Excluding combination: $video, $scene_number, $width, $height, $square_size, $h_stride, $v_stride, $neighbor, $ref, $subvideo"
                                            continue
                                        fi

                                        # Call the orchestrator script with the combination, suppressing its output
                                        echo "Running orchestrator with parameters: $video, $scene_number, $width, $height, $square_size, $h_stride, $v_stride, $neighbor, $ref, $subvideo"
                                        ./orchestrator.sh "$video" "$scene_number" "$width" "$height" "$square_size" "$h_stride" "$v_stride" "$neighbor" "$ref" "$subvideo" > /dev/null 2>&1
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
