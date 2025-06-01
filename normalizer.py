import cv2
import numpy as np
import json

def generate_geojson_from_dict(image_dict, corners, geojson_data, color_map, epsilon_factor=0.001, contour_area_threshold=10):
    """
    Populate an existing GeoJSON-like dictionary with polygons from the class-specific images.

    Parameters:
    - image_dict: Dictionary where keys are class names and values are single image arrays (normalized RGB, numpy).
    - corners: List of corner coordinates for geo-referencing (top-left, top-right, bottom-right, bottom-left).
    - geojson_data: Predefined GeoJSON structure to populate with features.
    - color_map: Dictionary mapping class names to normalized RGB color tuples.
    - epsilon_factor: Factor for polygon simplification.
    - contour_area_threshold: Minimum area for contours to be considered.

    Returns:
    - geojson_data: Updated GeoJSON-like dictionary.
    """
    height, width, _ = next(iter(image_dict.values())).shape  # Assuming all images have the same shape
    top_left, top_right, bottom_right, bottom_left = corners

    # Functions for coordinate transformations
    def pixel_to_geo(pixel):
        return (
            float(top_left[0] + (pixel[0] / width) * (top_right[0] - top_left[0])),
            float(top_left[1] + (pixel[1] / height) * (bottom_left[1] - top_left[1]))
        )

    # Process each class and its corresponding image
    for class_name, image in image_dict.items():
        # Get the class color from the predefined color map (normalized RGB to 0-255)
        class_color = color_map.get(class_name)
        if class_color is None:
            print(f"Error: No color mapping found for class '{class_name}'. Skipping.")
            continue
        # Convert normalized color to 0-255 range
        class_color = tuple(int(c * 255) for c in class_color)

        # Convert normalized image to 0-255 range
        image = (image * 255).astype(np.uint8)

        # Convert image to a binary mask based on class color
        mask = cv2.inRange(image, class_color, class_color)
        binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Find contours for the current class in the image
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < contour_area_threshold:
                continue

            # Simplify the contour
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            # Convert pixel coordinates to geographic coordinates
            coordinates = [pixel_to_geo(tuple(pt[0])) for pt in polygon]

            # Ensure the polygon is closed
            if coordinates and coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            # Create a GeoJSON feature for the current class
            feature = {
                'type': 'Feature',
                'properties': {'class': class_name},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coordinates]
                }
            }
            geojson_data['features'].append(feature)

    return geojson_data


# Example Usage
if __name__ == "__main__":
    # Define input dictionary where each key is a class, and each value is a single normalized RGB image
    image_dict = {
        'urban': np.ones((100, 100, 3), dtype=np.float32) * [1.0, 0.0, 0.0],  # Red
        'water': np.ones((100, 100, 3), dtype=np.float32) * [0.0, 0.0, 1.0],  # Blue
        'forest': np.ones((100, 100, 3), dtype=np.float32) * [0.0, 1.0, 0.0]  # Green
    }

    # Define normalized color map for the classes
    color_map = {
        'urban': (1.0, 0.0, 0.0),    # Red
        'water': (0.0, 0.0, 1.0),    # Blue
        'forest': (0.0, 1.0, 0.0)    # Green
    }

    # Define dynamic corner coordinates for geo-referencing
    corners = [
        (77.64210216899312, 29.519690553168758),  # Top-left (lat, lon)
        (77.79366733198111, 29.519690553168758),  # Top-right
        (77.79366733198111, 29.408070020352426),  # Bottom-right
        (77.64210216899312, 29.408070020352426)   # Bottom-left
    ]

    # Predefined GeoJSON structure
    geojson_data = {
        'type': 'FeatureCollection',
        'features': []
    }

    # Generate the GeoJSON dictionary dynamically and update it
    try:
        geojson_data = generate_geojson_from_dict(image_dict, corners, geojson_data, color_map)
        # Print the updated GeoJSON-like dictionary
        print(json.dumps(geojson_data, indent=4))
    except ValueError as e:
        print(f"Error: {e}")