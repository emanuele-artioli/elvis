import numpy as np
import sys

# Dictionary representing the bitrate ladder (bitrate: pixels)
bitrate_ladder = {
    0: 0,
    145 * 0.8: (240 * 4/3) * 240,
    240 * 0.8: (240 * 16/9) * 240,
    365 * 0.8: (360 * 4/3) * 360,
    500 * 0.8: (360 * 4/3) * 360,
    600 * 0.8: (360 * 16/9) * 360,
    750 * 0.8: (360 * 16/9) * 360,
    900 * 0.8: (360 * 16/9) * 360,
    1000 * 0.8: (480 * 4/3) * 480,
    1100 * 0.8: (480 * 16/9) * 480,
    1200 * 0.8: (540 * 4/3) * 540,
    1400 * 0.8: (540 * 16/9) * 540,
    1600 * 0.8: (720 * 4/3) * 720,
    1800 * 0.8: (720 * 4/3) * 720,
    2000 * 0.8: (720 * 4/3) * 720,
    2250 * 0.8: (720 * 4/3) * 720,
    2500 * 0.8: (720 * 4/3) * 720,
    2800 * 0.8: (1080 * 4/3) * 1080,
    3000 * 0.8: (720 * 16/9) * 720,
    3200 * 0.8: (720 * 16/9) * 720,
    3400 * 0.8: (720 * 16/9) * 720,
    3750 * 0.8: (720 * 16/9) * 720,
    4000 * 0.8: (1080 * 4/3) * 1080,
    4300 * 0.8: (1080 * 4/3) * 1080,
    4500 * 0.8: (1080 * 4/3) * 1080,
    5000 * 0.8: (1080 * 16/9) * 1080,
    5500 * 0.8: (1080 * 16/9) * 1080,
    6000 * 0.8: (1080 * 16/9) * 1080,
    6500 * 0.8: (1080 * 16/9) * 1080,
    7000 * 0.8: (1080 * 16/9) * 1080,
}

def calculate_bitrate(width, height):
    # Calculate the number of pixels for the input resolution
    num_pixels = width * height
    
    # Extract the keys and values from the bitrate ladder dictionary
    bitrates = np.array(list(bitrate_ladder.keys()))
    pixels = np.array(list(bitrate_ladder.values()))
    
    # Fit a polynomial curve of degree 2 (quadratic) to the data
    polynomial_coefficients = np.polyfit(pixels, bitrates, 2)
    polynomial = np.poly1d(polynomial_coefficients)
    
    # Calculate and return the bitrate for the input number of pixels
    bitrate = polynomial(num_pixels)
    return int(bitrate)

def main():
    if len(sys.argv) != 3:
        print("Usage: python bitrate_calculator.py <width> <height>")
        sys.exit(1)
    
    width = int(sys.argv[1])
    height = int(sys.argv[2])
    
    bitrate = calculate_bitrate(width, height)
    # ffmpeg needs the bitrate in bps, not kbps
    print(bitrate * 1000)

if __name__ == "__main__":
    main()
