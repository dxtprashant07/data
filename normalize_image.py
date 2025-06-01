import numpy as np

def normalize_rgb_array(rgb_array):
    """
    Normalize an RGB array with values in the range [0, 255] to [0, 1].

    Parameters:
    - rgb_array: numpy.ndarray, the input RGB array with values in the range [0, 255].

    Returns:
    - normalized_rgb: numpy.ndarray, the normalized RGB array with values in the range [0, 1].
    """
    # Ensure the input is a NumPy array
    if not isinstance(rgb_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    # Check if the values are within the range [0, 255]
    if rgb_array.min() < 0 or rgb_array.max() > 255:
        raise ValueError("RGB values must be in the range [0, 255].")

    # Normalize to the range [0, 1]
    normalized_rgb = rgb_array / 255.0

    return normalized_rgb


# Example RGB array
rgb_array = np.array([[128, 169, 32],[255, 200, 50],
    [0, 100, 220]])  # Example RGB values
normalized_rgb = normalize_rgb_array(rgb_array)

print("Original RGB Array:", rgb_array)
print("Normalized RGB Array:", normalized_rgb)
