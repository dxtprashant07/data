import cv2
import numpy as np
import json

def generate_geojson_from_dict(image_dict, corners, geojson_data, color_map, epsilon_factor=0.001, contour_area_threshold=10):
    """
    Populate an existing GeoJSON-like dictionary with polygons from the class-specific images.

    Parameters:
    - image_dict: Dictionary where keys are class names and values are lists of image arrays (numpy).
    - corners: List of corner coordinates for geo-referencing (top-left, top-right, bottom-right, bottom-left).
    - geojson_data: Predefined GeoJSON structure to populate with features.
    - color_map: Dictionary mapping class names to BGR color tuples.
    - epsilon_factor: Factor for polygon simplification.
    - contour_area_threshold: Minimum area for contours to be considered.

    Returns:
    - geojson_data: Updated GeoJSON-like dictionary.
    """
    height, width, _ = next(iter(image_dict.values()))[0].shape  # Assuming all images have the same shape, and each class has multiple images
    top_left, top_right, bottom_right, bottom_left = corners

    # Functions for coordinate transformations (adjust to match dynamic corners)
    pixel_to_geo = lambda pixel: (
        float(top_left[0] + (pixel[0] / width) * (top_right[0] - top_left[0])),
        float(top_left[1] + (pixel[1] / height) * (bottom_left[1] - top_left[1]))
    )

    # Process each class and its corresponding images
    for class_name, images in image_dict.items():
        # Get the class color from the predefined color map
        class_color = color_map.get(class_name)
        if class_color is None:
            print(f"Error: No color mapping found for class '{class_name}'. Skipping.")
            continue

        # Process each image in the class's image list
        for image in images:
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
    # Define input dictionary where each key is a class, and each value is a list of image arrays
    image_dict = {
        'urban': [cv2.imread('urban_class_1.png'), cv2.imread('urban_class_2.png')],  # Replace with actual image paths
        'water': [cv2.imread('water_class_1.png'), cv2.imread('water_class_2.png')],
        'forest': [cv2.imread('forest_class_1.png'), cv2.imread('forest_class_2.png')]
    }

    # Validate that all images are loaded
    for class_name, images in image_dict.items():
        for img in images:
            if img is None:
                print(f"Error: Image for class '{class_name}' is missing or invalid.")
                exit(1)

    # Define dynamic corner coordinates for geo-referencing (this can be user-defined or calculated dynamically)
    corners = [
        (77.64210216899312, 29.519690553168758),  # Top-left (lat, lon)
        (77.79366733198111, 29.519690553168758),  # Top-right
        (77.79366733198111, 29.408070020352426),  # Bottom-right
        (77.64210216899312, 29.408070020352426)   # Bottom-left
    ]

    # Predefined GeoJSON structure (this is an example, your predefined structure might be different)
    geojson_data = {
        'type': 'FeatureCollection',
        'features': []
    }

    # Predefined color map for the classes
    color_map = {
        'urban': (0, 0, 255),    # Red for urban areas (BGR)
        'water': (255, 0, 0),    # Blue for water bodies (BGR)
        'forest': (0, 255, 0)    # Green for forest (BGR)
    }

    # Generate the GeoJSON dictionary dynamically and update it
    try:
        geojson_data = generate_geojson_from_dict(image_dict, corners, geojson_data, color_map)
        # Print the updated GeoJSON-like dictionary
        print(json.dumps(geojson_data, indent=4))
    except ValueError as e:
        print(f"Error: {e}")
