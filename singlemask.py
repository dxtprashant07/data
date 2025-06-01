import cv2
import numpy as np
import json

def process_dynamic_image(image, color_bgr, corners, epsilon_factor=0.001, contour_area_threshold=10):
    """
    Process a dynamically provided image for a specific color to generate GeoJSON data.
    """
    # Geo-referencing corners
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

    if not all_polygons:
        raise ValueError("No valid polygons found in the image.")

    # Create GeoJSON data
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[poly] for poly in all_polygons]
                }
            }
        ]
    }

    return geojson_data


# Example Usage
if __name__ == "__main__":
    # Dynamically load an image from file
    image_path = 'data/class_masks/Andaman_and_Nicobar/water/5_0_water_mask.png'  # Replace with your image file path
    image = cv2.imread(image_path)  # Load the image as a NumPy array

    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # Dynamic coordinates for geo-referencing
        corners = [
            (77.64210216899312, 29.519690553168758),  # Top-left
            (77.79366733198111, 29.519690553168758),  # Top-right
            (77.79366733198111, 29.408070020352426),  # Bottom-right
            (77.64210216899312, 29.408070020352426),  # Bottom-left
        ]

        # Dynamic color (BGR format)
        color_bgr = (223, 155, 65)  # Replace with the color you want to find (e.g., water color)

        try:
            geojson_result = process_dynamic_image(image, color_bgr, corners)
            print("Generated GeoJSON:")
            print(json.dumps(geojson_result, indent=4))
        except ValueError as e:
            print(f"Error: {e}")
