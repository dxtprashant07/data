import os
import re
import cv2
import numpy as np

def find_mask_file(mask_folder, image_file):
    """
    Find a matching mask file for a given image file in the mask folder.
    Matches numbers in filenames (e.g., "image_5" matches "mask_5").

    Parameters:
        mask_folder (str): Path to the folder containing masks.
        image_file (str): The image file name.

    Returns:
        str: Path to the matching mask file, or None if not found.
    """
    # Extract the number from the image file name
    image_number = re.search(r'\d+', image_file)
    if not image_number:
        return None
    image_number = image_number.group()

    # Search for a mask with the same number
    for mask_file in os.listdir(mask_folder):
        mask_number = re.search(r'\d+', mask_file)
        if mask_number and mask_number.group() == image_number:
            return os.path.join(mask_folder, mask_file)

    return None

def apply_mask_with_transparency(image_path, mask_path, output_folder):
    """
    Apply a mask to an image, making missing parts transparent and saving in a subfolder.

    Parameters:
        image_path (str): Path to the input image.
        mask_path (str): Path to the masking image.
        output_folder (str): Path to save the output image.
    """
    # Load the input image and mask
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error: Image or mask could not be loaded for {image_path} or {mask_path}.")
        return

    # Resize the mask to match the image dimensions
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Normalize the mask to be binary (0 and 255)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Create an alpha channel (0 for missing pixels, 255 for existing pixels)
    alpha_channel = binary_mask

    # Merge the image with the alpha channel
    if image.shape[2] == 3:  # If image has no alpha channel
        image_with_alpha = cv2.merge([image, alpha_channel])
    elif image.shape[2] == 4:  # If image already has an alpha channel
        image_with_alpha = image.copy()
        image_with_alpha[:, :, 3] = alpha_channel

    # Create a unique subfolder for the output
    base_filename = os.path.splitext(os.path.basename(image_path))[0]  # Get the file name without extension
    subfolder = os.path.join(output_folder, base_filename)
    os.makedirs(subfolder, exist_ok=True)

    # Define the output path
    output_path = os.path.join(subfolder, f"{base_filename}_output.png")

    # Save the resulting image with transparency
    cv2.imwrite(output_path, image_with_alpha)
    print(f"Output saved with transparency at {output_path}")

def process_folder(image_folder, mask_folder, output_folder):
    """
    Process all images in the image folder and apply the corresponding masks,
    saving each output in a unique subfolder.

    Parameters:
        image_folder (str): Path to the folder containing images.
        mask_folder (str): Path to the folder containing masks.
        output_folder (str): Path to save the output images.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_path = find_mask_file(mask_folder, image_file)

        print(f"Processing: {image_path} with mask: {mask_path}")

        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            continue
        if mask_path is None or not os.path.exists(mask_path):
            print(f"Error: Mask file not found for {image_file}")
            continue

        try:
            apply_mask_with_transparency(image_path, mask_path, output_folder)
        except Exception as e:
            print(f"Error processing {image_path} with {mask_path}: {e}")

# Example usage
if __name__ == "__main__":
    image_folder = "train_image"
    mask_folder = "train_mask"
    output_folder = "output"
    process_folder(image_folder, mask_folder, output_folder)