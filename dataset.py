import cv2
import numpy as np


def correct_tilt(image_path, output_path):
    """
    Correct the tilt of a single image by detecting its dominant orientation and rotating it.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the corrected image.
    """
    try:
        # Load the image in grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            print("No lines detected; skipping tilt correction.")
            return

        # Calculate the angles of the detected lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90  # Convert from radians to degrees
            angles.append(angle)

        # Calculate the median angle
        median_angle = np.median(angles)

        # Rotate the image by the negative of the median angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Save the corrected image
        cv2.imwrite(output_path, rotated_image)
        print(f"Tilt corrected and saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    # Path to the tilted image
    image_path = "gettyimages.jpg"  # Replace with your image's path
    output_path = "corrected_image.jpg"  # Path to save the corrected image

    # Correct the tilt of the image
    correct_tilt(image_path, output_path)
