import cv2
import numpy as np


def main():
    # Load the satellite image
    image_path = 'area_2.png'  # Replace with your satellite image path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    # Get image dimensions
    height, width, _ = image.shape

    # Example: User-provided float coordinates for the four edges (in a custom coordinate system)
    # These should correspond to the bounds of the image
    user_top_left = (77.64210216899312, 29.519690553168758)  # Float coordinate of the top-left corner
    user_top_right = (77.79366733198111, 29.519690553168758)  # Float coordinate of the top-right corner
    user_bottom_right = (77.79366733198111, 29.408070020352426)  # Float coordinate of the bottom-right corner
    user_bottom_left = (77.64210216899312, 29.408070020352426)  # Float coordinate of the bottom-left corner

    # Function to map pixel indices to float coordinates
    def map_to_float_coords(x, y):
        float_x = user_top_left[0] + (user_top_right[0] - user_top_left[0]) * (x / (width - 1))
        float_y = user_top_left[1] + (user_bottom_left[1] - user_top_left[1]) * (y / (height - 1))
        return (float_x, float_y)

    # Array to store pixel coordinates
    float_pixel_coordinates = []

    # Store pixel coordinates in clockwise order (adjusted to float values)
    for x in range(width):  # Top row
        float_pixel_coordinates.append(map_to_float_coords(x, 0))

    for y in range(1, height - 1):  # Right column
        float_pixel_coordinates.append(map_to_float_coords(width - 1, y))

    for x in range(width - 1, -1, -1):  # Bottom row
        float_pixel_coordinates.append(map_to_float_coords(x, height - 1))

    for y in range(height - 2, 0, -1):  # Left column
        float_pixel_coordinates.append(map_to_float_coords(0, y))

    # Store remaining layers in clockwise order (if needed)
    # for layer in range(1, (min(height, width) + 1) // 2):
    #     # Top row
    #     for x in range(layer, width - layer):
    #         float_pixel_coordinates.append(map_to_float_coords(x, layer))

    #     # Right column
    #     for y in range(layer, height - layer):
    #         float_pixel_coordinates.append(map_to_float_coords(width - layer - 1, y))

    #     # Bottom row
    #     for x in range(width - layer - 1, layer - 1, -1):
    #         float_pixel_coordinates.append(map_to_float_coords(x, height - layer - 1))

    #     # Left column
    #     for y in range(height - layer - 2, layer, -1):
    #         float_pixel_coordinates.append(map_to_float_coords(layer, y))

    # Print all stored pixel coordinates
    

    #print(f"Pixel coordinates: {float_pixel_coordinates}")
    print("Stored pixel coordinates (mapped to float) in clockwise order:")
    for coords in float_pixel_coordinates:
        print(coords)

    print(f"Total unique pixel coordinates stored: {len(float_pixel_coordinates)}")

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
