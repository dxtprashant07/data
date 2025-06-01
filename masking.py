import cv2
import os
import pandas as pd
import numpy as np


class ColorAnalyzer:
    def __init__(self, img_array, color_ranges):
        """
        Initialize the ColorAnalyzer class.

        Parameters:
        - img_array: The input image array (BGR format).
        - color_ranges: A dictionary of color ranges in BGR format.
        """
        self.img_array = img_array
        self.color_ranges = color_ranges

    def generate_single_color_masks(self):
        """
        Generate new mask images for each color, where only one color is preserved,
        and the rest of the mask is set to black.

        Returns:
        - A dictionary with color names as keys and the corresponding single-color masks as values.
        """
        single_color_masks = {}
        for color_name, bgr in self.color_ranges.items():
            lower_bound = np.array(bgr, dtype=np.uint8)
            upper_bound = np.array(bgr, dtype=np.uint8)

            # Create a binary mask for the current color
            mask = cv2.inRange(self.img_array, lower_bound, upper_bound)

            # Create a new mask image with only the current color and black background
            single_color_image = np.zeros_like(self.img_array)
            single_color_image[mask > 0] = bgr
            single_color_masks[color_name] = single_color_image

        return single_color_masks


def load_color_ranges_from_csv(csv_path):
    """
    Load color ranges from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing color definitions.

    Returns:
    - A dictionary with color names as keys and BGR values as tuples.
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        color_ranges = {}
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            color_name = row["name"]
            bgr = (int(row[" b"]), int(row[" g"]), int(row[" r"]))
            color_ranges[color_name] = bgr

        return color_ranges
    except Exception as e:
        print(f"Error loading color ranges from CSV: {e}")
        return {}


def save_single_color_masks_by_class(dataset_folder, color_ranges):
    """
    Save new mask images and corresponding images in class-specific subfolders.

    Parameters:
    - dataset_folder: Path to the dataset folder.
    - color_ranges: A dictionary of color ranges.
    """
    for folder_type in ["train", "test", "val"]:
        image_folder = os.path.join(dataset_folder, f"{folder_type}_image")
        mask_folder = os.path.join(dataset_folder, f"{folder_type}_mask")
        class_image_folder = os.path.join(dataset_folder, f"{folder_type}_class_image")
        class_mask_folder = os.path.join(dataset_folder, f"{folder_type}_class_mask")

        if not os.path.exists(mask_folder):
            print(f"Skipping {folder_type}: Missing mask folder.")
            continue

        os.makedirs(class_image_folder, exist_ok=True)
        os.makedirs(class_mask_folder, exist_ok=True)

        print(f"Processing {folder_type} masks...")

        # Iterate through the mask images in the mask folder
        for img_file in os.listdir(mask_folder):
            img_path = os.path.join(mask_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading mask image: {img_file}")
                continue

            # Generate single-color masks for each color in the mask
            analyzer = ColorAnalyzer(img, color_ranges)
            single_color_masks = analyzer.generate_single_color_masks()

            # Save each single-color mask and corresponding image
            for class_name, single_color_image in single_color_masks.items():
                # Create class-specific subfolders
                class_image_subfolder = os.path.join(class_image_folder, class_name)
                class_mask_subfolder = os.path.join(class_mask_folder, class_name)
                os.makedirs(class_image_subfolder, exist_ok=True)
                os.makedirs(class_mask_subfolder, exist_ok=True)

                # Save the single-color mask
                mask_output_path = os.path.join(
                    class_mask_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_mask.png"
                )
                cv2.imwrite(mask_output_path, single_color_image)

                # Save the corresponding image
                original_image_path = os.path.join(image_folder, img_file)
                if os.path.exists(original_image_path):
                    original_image = cv2.imread(original_image_path)
                    matched_image_output_path = os.path.join(
                        class_image_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_image.png"
                    )
                    cv2.imwrite(matched_image_output_path, original_image)

        print(f"Saved class-specific masks in: {class_mask_folder}")
        print(f"Saved class-specific images in: {class_image_folder}")


if __name__ == "__main__":
    # Define paths
    dataset_folder = r"C:\Users\prash\OneDrive\Desktop\dataset"
    csv_path = r"C:\Users\prash\OneDrive\Desktop\dataset\class_dict_seg.csv"

    # Load color ranges
    color_ranges = load_color_ranges_from_csv(csv_path)

    if not color_ranges:
        print("No color ranges loaded. Exiting.")
    else:
        save_single_color_masks_by_class(dataset_folder, color_ranges)
