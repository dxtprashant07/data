import os
import cv2
import numpy as np
import json
import pandas as pd


def process_dataset(dataset_folder, csv_path, geojson_output_folder, lower_gray, upper_gray, corners, epsilon_factor=0.001, contour_area_threshold=10):
    """
    Process the dataset to generate class-specific masks, images, and GeoJSON files.

    Parameters:
    - dataset_folder: Path to the dataset containing images and masks.
    - csv_path: Path to the CSV file containing color ranges.
    - geojson_output_folder: Folder to save GeoJSON files.
    - lower_gray: Lower bound for grayscale threshold.
    - upper_gray: Upper bound for grayscale threshold.
    - corners: Coordinates for geo-referencing the images.
    - epsilon_factor: Factor for polygon simplification.
    - contour_area_threshold: Minimum area for contours to be considered.
    """
    # Validate input paths
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder not found: {dataset_folder}")
        return
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load color ranges
    print("Loading color ranges from CSV...")
    color_ranges = load_color_ranges_from_csv(csv_path)
    if not color_ranges:
        print("No color ranges loaded. Exiting.")
        return

    # Step 1: Generate class-specific masks and images
    print("Generating class-specific masks and images...")
    save_single_color_masks_by_class(dataset_folder, color_ranges)

    # Step 2: Generate GeoJSON files from masks
    print("Generating GeoJSON files from masks...")
    class_mask_folder = os.path.join(dataset_folder, "class_masks")
    process_images_in_folders(class_mask_folder, geojson_output_folder, lower_gray, upper_gray, corners, epsilon_factor, contour_area_threshold)

    print("Processing completed!")


def load_color_ranges_from_csv(csv_path):
    """
    Load color ranges from a CSV file.
    """
    color_ranges = {}
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        color_name = row["name"]
        bgr = (int(row[" b"]), int(row[" g"]), int(row[" r"]))
        color_ranges[color_name] = bgr

    return color_ranges


def save_single_color_masks_by_class(dataset_folder, color_ranges):
    """
    Save new mask images and corresponding images in class-specific subfolders.
    """
    image_folder = os.path.join(dataset_folder, "images")
    mask_folder = os.path.join(dataset_folder, "masks")
    class_image_folder = os.path.join(dataset_folder, "class_images")
    class_mask_folder = os.path.join(dataset_folder, "class_masks")

    state_folders = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

    for state in state_folders:
        state_image_folder = os.path.join(image_folder, state)
        state_mask_folder = os.path.join(mask_folder, state)
        state_class_image_folder = os.path.join(class_image_folder, state)
        state_class_mask_folder = os.path.join(class_mask_folder, state)

        os.makedirs(state_class_image_folder, exist_ok=True)
        os.makedirs(state_class_mask_folder, exist_ok=True)

        for img_file in os.listdir(state_mask_folder):
            img_path = os.path.join(state_mask_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            analyzer = ColorAnalyzer(img, color_ranges)
            single_color_masks = analyzer.generate_single_color_masks()

            for class_name, single_color_image in single_color_masks.items():
                class_image_subfolder = os.path.join(state_class_image_folder, class_name)
                class_mask_subfolder = os.path.join(state_class_mask_folder, class_name)

                os.makedirs(class_image_subfolder, exist_ok=True)
                os.makedirs(class_mask_subfolder, exist_ok=True)

                mask_output_path = os.path.join(class_mask_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_mask.png")
                cv2.imwrite(mask_output_path, single_color_image)

                mask_filename_without_ext = os.path.splitext(img_file)[0]
                image_candidates = [f for f in os.listdir(state_image_folder) if os.path.splitext(f)[0] == mask_filename_without_ext]

                for img_candidate in image_candidates:
                    original_image_path = os.path.join(state_image_folder, img_candidate)
                    if os.path.exists(original_image_path):
                        original_image = cv2.imread(original_image_path)
                        if original_image is not None:
                            matched_image_output_path = os.path.join(class_image_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_image.png")
                            cv2.imwrite(matched_image_output_path, original_image)


def process_images_in_folders(input_folder, output_folder, lower_bound, upper_bound, corners, epsilon_factor=0.0005, contour_area_threshold=50):
    """
    Process all mask images in a folder recursively, saving their polygons as JSON files in a separate folder.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                geojson_output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(geojson_output_dir, exist_ok=True)

                draw_polygons_and_save_geojson(image_path, lower_bound, upper_bound, corners, geojson_output_dir, epsilon_factor, contour_area_threshold)


def draw_polygons_and_save_geojson(image_path, lower_bound, upper_bound, corners, output_dir, epsilon_factor=0.0005, contour_area_threshold=50):
    """
    Process an image, group all polygons into one GeoJSON feature, and save as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return

    height, width = image.shape

    top_left, top_right, bottom_right, bottom_left = corners
    geo_to_pixel = lambda coord: (
        int(((coord[0] - top_left[0]) / (top_right[0] - top_left[0])) * width),
        int(((coord[1] - top_left[1]) / (bottom_left[1] - top_left[1])) * height)
    )
    pixel_to_geo = lambda pixel: (
        float(top_left[0] + (pixel[0] / width) * (top_right[0] - top_left[0])),
        float(top_left[1] + (pixel[1] / height) * (bottom_left[1] - top_left[1]))
    )

    mask = cv2.inRange(cv2.GaussianBlur(image, (7, 7), 0), lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < contour_area_threshold:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        vertices = [pixel_to_geo(tuple(pt[0])) for pt in polygon]

        if vertices and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])

        all_polygons.append(vertices)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"image_name": os.path.basename(image_path)},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[poly] for poly in all_polygons]
                }
            }
        ]
    }

    geojson_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=4)


class ColorAnalyzer:
    def __init__(self, img_array, color_ranges):
        self.img_array = img_array
        self.color_ranges = color_ranges

    def generate_single_color_masks(self):
        single_color_masks = {}
        for color_name, bgr in self.color_ranges.items():
            lower_bound = np.array(bgr, dtype=np.uint8)
            upper_bound = np.array(bgr, dtype=np.uint8)

            mask = cv2.inRange(self.img_array, lower_bound, upper_bound)
            single_color_image = np.zeros_like(self.img_array)
            single_color_image[mask > 0] = bgr
            single_color_masks[color_name] = single_color_image

        return single_color_masks
