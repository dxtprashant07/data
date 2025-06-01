import cv2
import numpy as np
import json
import os

def process_dynamic_images(images, color_info, corners_list, epsilon_factor=0.001, contour_area_threshold=10):
    """
    Process dynamically provided images for specific colors to generate GeoJSON data in memory.
    
    Parameters:
    - images: List of image paths (strings) to process.
    - color_info: List of dictionaries containing 'color_bgr' (BGR color) and 'color_name' (color label).
    - corners_list: List of corner coordinates for geo-referencing each image.
    - epsilon_factor: Factor for polygon simplification.
    - contour_area_threshold: Minimum area for contours to be considered.
    
    Returns:
    - color_to_features: Dictionary containing GeoJSON data for each color.
    """
    color_to_features = {color['color_name']: [] for color in color_info}

    # Process each image
    for image_path, corners, color_info_item in zip(images, corners_list, color_info):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        height, width, _ = image.shape
        top_left, top_right, bottom_right, bottom_left = corners

        geo_to_pixel = lambda coord: (
            int(((coord[0] - top_left[0]) / (top_right[0] - top_left[0])) * width),
            int(((coord[1] - top_left[1]) / (bottom_left[1] - top_left[1])) * height)
        )
        pixel_to_geo = lambda pixel: (
            float(top_left[0] + (pixel[0] / width) * (top_right[0] - top_left[0])),
            float(top_left[1] + (pixel[1] / height) * (bottom_left[1] - top_left[1]))
        )

        color_bgr = color_info_item['color_bgr']
        color_name = color_info_item['color_name']

        # Create a mask for the specified color
        lower_bound = np.array(color_bgr, dtype=np.uint8)
        upper_bound = np.array(color_bgr, dtype=np.uint8)
        color_mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find contours in the color mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < contour_area_threshold:
                continue

            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            vertices = [pixel_to_geo(tuple(pt[0])) for pt in polygon]

            # Ensure the polygon is closed
            if vertices and vertices[0] != vertices[-1]:
                vertices.append(vertices[0])

            all_polygons.append(vertices)

        if all_polygons:
            geojson_feature = {
                "type": "Feature",
                "properties": {"color": color_name, "image_id": os.path.basename(image_path)},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[poly] for poly in all_polygons]
                }
            }

            color_to_features[color_name].append(geojson_feature)

    # Create nested GeoJSON structure
    geojson_data = {
        color_name: {
            "type": "FeatureCollection",
            "features": features
        } for color_name, features in color_to_features.items()
    }

    return geojson_data


# Example Usage
if __name__ == "__main__":
    # Paths to multiple images
    image_paths = [
        'data/class_masks/Andaman_and_Nicobar/water/4_2_water_mask.png',
        'data/class_masks/Andaman_and_Nicobar/water/3_1_water_mask.png',
        'data/class_masks/Andaman_and_Nicobar/trees/3_0_trees_mask.png',
        'data/class_masks/Andaman_and_Nicobar/trees/3_1_trees_mask.png'
    ]

    # Corner coordinates for geo-referencing
    corners_list = [
        [(92.7265, 11.6392), (92.7645, 11.6392), (92.7645, 11.6173), (92.7265, 11.6173)],  # Image 1
        [(92.7310, 11.6450), (92.7690, 11.6450), (92.7690, 11.6230), (92.7310, 11.6230)],  # Image 2
        [(76.2148, 9.9312), (76.2850, 9.9312), (76.2850, 9.8915), (76.2148, 9.8915)],    # Image 3
        [(76.2100, 9.9350), (76.2800, 9.9350), (76.2800, 9.8950), (76.2100, 9.8950)]     # Image 4
    ]

    # Colors and associated names (BGR format)
    color_info = [
        {'color_bgr': (223, 155, 65), 'color_name': 'water'},
        {'color_bgr': (223, 155, 65), 'color_name': 'water'},
        {'color_bgr': (73, 125, 57), 'color_name': 'trees'},
        {'color_bgr': (73, 125, 57), 'color_name': 'trees'}
    ]

    try:
        geojson_result = process_dynamic_images(image_paths, color_info, corners_list)
        # Print the GeoJSON result for inspection
        print(json.dumps(geojson_result, indent=4))
    except ValueError as e:
        print(f"Error: {e}")
