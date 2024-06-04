#!/bin/bash

# Define the combinations
videos=("Tears_of_Steel_4k" "bbb_sunflower_2160p_60fps_normal")
scene_numbers=("009")
widths=("1280")
heights=("720")
framerates=("24" "60")
square_sizes=("40")
horizontal_stride=("1" "2" "3")
vertical_stride=("1" "2" "3")
neighbor_length=("2" "5")
ref_stride=("2" "5")
subvideo_length=("10" "30")
bitrates=("1000k" "3000k")

# Function to check if a combination should be excluded
should_exclude_combination() {
    local video="$1"
    local scene_number="$2"
    local width="$3"
    local height="$4"
    local framerate="$5"
    local square_size="$6"
    local h_stride="$7"
    local v_stride="$8"
    local neighbor="$9"
    local ref="${10}"
    local subvideo="${11}"
    local bitrate="${12}"

    # Add more exclusion rules as needed
    # if [[ <condition> ]]; then
    #     return 0  # Exclude this combination
    # fi

    if [[ "$width" == "640" && "$height" != "360" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$width" == "1280" && "$height" != "720" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$width" == "1920" && "$height" != "1080" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$video" == "Tears_of_Steel_1080p" && "$framerate" != "24" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$video" == "Tears_of_Steel_4k" && "$framerate" != "24" ]]; then
        return 0  # Exclude this combination
    fi

    if [[ "$video" == "bbb_sunflower_2160p_60fps_normal" && "$framerate" != "60" ]]; then
        return 0  # Exclude this combination
    fi

    # TODO: let's keep it to see if it actually ends up in the same video as original
    # if [[ "$h_stride" == "1" && "$v_stride" == "1" ]]; then
    #     return 0  # Exclude this combination
    # fi

    return 1  # Do not exclude this combination
}

# Iterate over each element of the arrays to generate combinations
for video in "${videos[@]}"; do
    for scene_number in "${scene_numbers[@]}"; do
        for width in "${widths[@]}"; do
            for height in "${heights[@]}"; do
                for framerate in "${framerates[@]}"; do
                    for square_size in "${square_sizes[@]}"; do
                        for h_stride in "${horizontal_stride[@]}"; do
                            for v_stride in "${vertical_stride[@]}"; do
                                for neighbor in "${neighbor_length[@]}"; do
                                    for ref in "${ref_stride[@]}"; do
                                        for subvideo in "${subvideo_length[@]}"; do
                                            for bitrate in "${bitrates[@]}"; do
                                                # Check if this combination should be excluded
                                                if should_exclude_combination "$video" "$scene_number" "$width" "$height" "$framerate" "$square_size" "$h_stride" "$v_stride" "$neighbor" "$ref" "$subvideo" "$bitrate"; then
                                                    echo "Excluding combination: $video, $scene_number, $width, $height, $framerate, $square_size, $h_stride, $v_stride, $neighbor, $ref, $subvideo, $bitrate"
                                                    continue
                                                fi

                                                # Call the orchestrator script with the combination, suppressing its output
                                                echo "Running orchestrator with parameters: $video, $scene_number, $width, $height, $framerate, $square_size, $h_stride, $v_stride, $neighbor, $ref, $subvideo, $bitrate"
                                                ./orchestrator.sh "$video" "$scene_number" "$width" "$height" "$framerate" "$square_size" "$h_stride" "$v_stride" "$neighbor" "$ref" "$subvideo" "$bitrate" > /dev/null 2>&1
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
done
