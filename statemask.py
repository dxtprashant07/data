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

        # Read each row and add the color name and BGR values to the dictionary
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
    image_folder = os.path.join(dataset_folder, "images")
    mask_folder = os.path.join(dataset_folder, "masks")
    class_image_folder = os.path.join(dataset_folder, "class_images")
    class_mask_folder = os.path.join(dataset_folder, "class_masks")

    # Get state folders from the image folder
    state_folders = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

    for state in state_folders:
        state_image_folder = os.path.join(image_folder, state)
        state_mask_folder = os.path.join(mask_folder, state)
        state_class_image_folder = os.path.join(class_image_folder, state)
        state_class_mask_folder = os.path.join(class_mask_folder, state)

        # Create subfolders if they don't exist
        os.makedirs(state_class_image_folder, exist_ok=True)
        os.makedirs(state_class_mask_folder, exist_ok=True)

        # Iterate over the mask images in the state folder
        for img_file in os.listdir(state_mask_folder):
            img_path = os.path.join(state_mask_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if mask image cannot be loaded

            # Generate single-color masks for each color in the mask
            analyzer = ColorAnalyzer(img, color_ranges)
            single_color_masks = analyzer.generate_single_color_masks()

            # Iterate through each class-specific mask and save them
            for class_name, single_color_image in single_color_masks.items():
                class_image_subfolder = os.path.join(state_class_image_folder, class_name)
                class_mask_subfolder = os.path.join(state_class_mask_folder, class_name)

                # Create subfolders for the class if they don't exist
                os.makedirs(class_image_subfolder, exist_ok=True)
                os.makedirs(class_mask_subfolder, exist_ok=True)

                # Save the single-color mask with the same name as the original mask file
                mask_output_path = os.path.join(class_mask_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_image.png")
                cv2.imwrite(mask_output_path, single_color_image)

                # Normalize the image and mask filenames (excluding extensions) for matching
                mask_filename_without_ext = os.path.splitext(img_file)[0]
                image_candidates = [f for f in os.listdir(state_image_folder) if os.path.splitext(f)[0] == mask_filename_without_ext]
                
                # Iterate over image candidates that match the mask filename
                for img_candidate in image_candidates:
                    original_image_path = os.path.join(state_image_folder, img_candidate)
                    if os.path.exists(original_image_path):
                        original_image = cv2.imread(original_image_path)
                        if original_image is not None:
                            # Save the corresponding image with the same name as the mask file
                            matched_image_output_path = os.path.join(class_image_subfolder, f"{os.path.splitext(img_file)[0]}_{class_name}_image.png")
                            cv2.imwrite(matched_image_output_path, original_image)

        # Print completion messages for each state
        print(f"Saved class-specific masks in: {state_class_mask_folder}")
        print(f"Saved class-specific images in: {state_class_image_folder}")

if __name__ == "__main__":
    # Define dataset and CSV paths
    dataset_folder = r"C:\Users\prash\OneDrive\Desktop\data07\data"
    csv_path = r"C:\Users\prash\OneDrive\Desktop\data07\class_dict_seg.csv"

    # Load color ranges from CSV file
    color_ranges = load_color_ranges_from_csv(csv_path)

    # If color ranges are loaded successfully, proceed with saving the masks and images
    if color_ranges:
        save_single_color_masks_by_class(dataset_folder, color_ranges)
    else:
        print("No color ranges loaded. Exiting.")
