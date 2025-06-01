import cv2
import numpy as np
import json
import os


def draw_polygons_and_save_geojson(image_path, lower_bound, upper_bound, corners, output_dir, epsilon_factor=0.0005, contour_area_threshold=50):
    """
    Process an image, group all polygons into one GeoJSON feature, and save as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return

    height, width = image.shape

    # Geographic <-> Pixel coordinate mapping functions
    top_left, top_right, bottom_right, bottom_left = corners
    geo_to_pixel = lambda coord: (
        int(((coord[0] - top_left[0]) / (top_right[0] - top_left[0])) * width),
        int(((coord[1] - top_left[1]) / (bottom_left[1] - top_left[1])) * height)
    )
    pixel_to_geo = lambda pixel: (
        float(top_left[0] + (pixel[0] / width) * (top_right[0] - top_left[0])),
        float(top_left[1] + (pixel[1] / height) * (bottom_left[1] - top_left[1]))
    )

    # Preprocessing: Gaussian Blur and threshold
    mask = cv2.inRange(cv2.GaussianBlur(image, (7, 7), 0), lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < contour_area_threshold:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Convert pixel coordinates to geographic coordinates
        vertices = [pixel_to_geo(tuple(pt[0])) for pt in polygon]

        # Ensure the polygon is closed
        if vertices and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])  # Close the polygon

        all_polygons.append(vertices)

    # Create a single GeoJSON with all polygons
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"image_name": os.path.basename(image_path)},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[poly] for poly in all_polygons]  # Each polygon must be closed
                }
            }
        ]
    }

    # Save the GeoJSON
    geojson_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=4)


def process_images_in_folders(input_folder, output_folder, lower_bound, upper_bound, corners, epsilon_factor=0.0005, contour_area_threshold=50):
    """
    Process all mask images in a folder recursively, saving their polygons as JSON files in a separate folder.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(root, file)

                # Create corresponding output directory in the 'geojson_output' folder
                relative_path = os.path.relpath(root, input_folder)
                geojson_output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(geojson_output_dir, exist_ok=True)

                draw_polygons_and_save_geojson(image_path, lower_bound, upper_bound, corners, geojson_output_dir, epsilon_factor, contour_area_threshold)


if __name__ == "__main__":
    # Folder containing generated masks
    input_folder = r"C:\Users\prash\OneDrive\Desktop\data07\data\class_masks"
    
    # Folder to store GeoJSON files (in a parallel structure)
    geojson_output_folder = r"C:\Users\prash\OneDrive\Desktop\data07\data\geojson_output"

    # Range of gray levels to consider for masks
    lower_gray, upper_gray = 100, 200

    # Geographic coordinates of the corners
    corners = [
        (77.64210216899312, 29.519690553168758),
        (77.79366733198111, 29.519690553168758),
        (77.79366733198111, 29.408070020352426),
        (77.64210216899312, 29.408070020352426)
    ]

    # Process all images in the input folder
    process_images_in_folders(input_folder, geojson_output_folder, lower_gray, upper_gray, corners, epsilon_factor=0.001, contour_area_threshold=10)
