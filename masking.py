import cv2
import os
import pandas as pd
import numpy as np


class ColorMask:
    def __init__(self, img_array, color_ranges):
        """
        Initialize the ColorMask class.

        Parameters:
        - img_array: The input image array (BGR format).
        - color_ranges: A dictionary of color ranges in BGR format.
        """
        self.img_array = img_array  # Original image in BGR format
        self.color_ranges = color_ranges  # Color ranges loaded from the CSV
        self.color_masks = {}  # Dictionary to store the generated masks

    def create_color_masks(self):
        """
        Create binary masks for each color range defined in the color_ranges dictionary.
        """
        for color_name, (bgr) in self.color_ranges.items():
            lower_bound = np.array(bgr, dtype=np.uint8)  # Lower BGR bounds
            upper_bound = np.array(bgr, dtype=np.uint8)  # Upper BGR bounds

            # Create a binary mask for the current color
            mask = cv2.inRange(self.img_array, lower_bound, upper_bound)

            # Store the mask
            self.color_masks[color_name] = mask

        return self.color_masks


def load_color_ranges_from_csv(csv_path):
    """
    Load color ranges from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing color definitions.

    Returns:
    - A dictionary with color names as keys and BGR values as tuples.
    """
    try:
        color_ranges = {}
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            color_name = row["name"]
            bgr = (int(row[" b"]), int(row[" g"]), int(row[" r"]))  # Convert RGB to BGR order
            color_ranges[color_name] = bgr

        return color_ranges
    except Exception as e:
        print(f"Error loading color ranges from CSV: {e}")
        return {}


def duplicate_images_to_match_masks(original_image_folder, mask_folder, output_folder):
    """
    Duplicate and rename original images to match the filenames of binary masks.

    Parameters:
    - original_image_folder: Path to the folder containing original images.
    - mask_folder: Path to the folder containing binary masks.
    - output_folder: Path to save the renamed original images.
    """
    try:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get a sorted list of mask files
        mask_files = sorted(os.listdir(mask_folder))

        for mask_file in mask_files:
            # Get the base name of the mask file (without extension)
            base_name = os.path.splitext(mask_file)[0]

            # Find a matching original image
            original_image_path = os.path.join(original_image_folder, base_name.split('_')[0] + ".png")

            if os.path.exists(original_image_path):
                # Save the original image with the mask filename in the output folder
                renamed_image_path = os.path.join(output_folder, mask_file)
                cv2.imwrite(renamed_image_path, cv2.imread(original_image_path))
                print(f"Duplicated original image: {original_image_path} -> {renamed_image_path}")
            else:
                print(f"No matching original image found for mask: {mask_file}")

    except Exception as e:
        print(f"Error duplicating images: {e}")


def process_images_and_masks(dataset_folder, color_ranges):
    """
    Process images and masks, creating binary masks and renaming/duplicating images to match them.

    Parameters:
    - dataset_folder: Path to the dataset folder.
    - color_ranges: A dictionary of color ranges.
    """
    for folder_type in ["train", "test", "val"]:
        # Paths for images and masks
        image_folder = os.path.join(dataset_folder, f"{folder_type}_image")
        mask_folder = os.path.join(dataset_folder, f"{folder_type}_mask")
        binary_mask_folder = os.path.join(dataset_folder, f"{folder_type}_mask_binary_masking")
        image_rename_folder = os.path.join(dataset_folder, f"{folder_type}_image_rename")

        if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
            print(f"Skipping {folder_type}: Missing image or mask folder.")
            continue

        print(f"Processing {folder_type}...")

        # Create binary masks
        os.makedirs(binary_mask_folder, exist_ok=True)
        for img_file in os.listdir(mask_folder):
            img_path = os.path.join(mask_folder, img_file)

            # Load the mask image
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image: {img_file}")
                continue

            # Create ColorMask object
            color_masker = ColorMask(img, color_ranges)

            # Generate color masks
            colored_masks = color_masker.create_color_masks()

            # Save each binary mask
            for color_name, mask in colored_masks.items():
                output_path = os.path.join(binary_mask_folder, f"{os.path.splitext(img_file)[0]}_{color_name}_mask.png")
                cv2.imwrite(output_path, mask)
                print(f"Saved binary mask: {output_path}")

        # Rename and duplicate images to match binary masks
        duplicate_images_to_match_masks(image_folder, binary_mask_folder, image_rename_folder)


if __name__ == "__main__":
    # Define paths
    dataset_folder = "dataset"
    csv_path = os.path.join(dataset_folder, "class_dict_seg.csv")

    # Load color ranges
    color_ranges = load_color_ranges_from_csv(csv_path)

    if not color_ranges:
        print("No color ranges loaded. Exiting.")
    else:
        process_images_and_masks(dataset_folder, color_ranges)
