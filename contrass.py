import cv2
import numpy as np
from normalizer import generate_geojson_from_dict

def increase_contrast_with_covariance(image, factor=1.5):
    """
    Increase the contrast of an image using covariance for color images.
    
    :param image: Input image (RGB).
    :param factor: Contrast scaling factor (greater than 1 for enhancement).
    :return: Contrast-enhanced image.
    """
    # Check if the image is RGB
    if len(image.shape) == 3:
        # Split the image into RGB channels
        b, g, r = cv2.split(image)
        # Stack channels to create a 2D array of pixel vectors
        pixel_matrix = np.stack((b.flatten(), g.flatten(), r.flatten()), axis=1)
        
        # Compute the covariance matrix of the pixel values
        covariance_matrix = np.cov(pixel_matrix, rowvar=False)
        
        # Calculate variance (diagonal elements of the covariance matrix)
        variances = np.diag(covariance_matrix)
        std_devs = np.sqrt(variances)
        
        # Scale each channel based on its standard deviation
        b = ((b - b.mean()) / std_devs[0]) * factor * std_devs[0] + b.mean()
        g = ((g - g.mean()) / std_devs[1]) * factor * std_devs[1] + g.mean()
        r = ((r - r.mean()) / std_devs[2]) * factor * std_devs[2] + r.mean()

        # Clip values to the valid range [0, 255]
        b = np.clip(b, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        r = np.clip(r, 0, 255).astype(np.uint8)
        
        # Merge channels back into an image
        enhanced_image = cv2.merge((b, g, r))
    else:
        # For grayscale, use the original mean and variance approach
        mean, std_dev = cv2.meanStdDev(image)
        mean = mean[0][0]
        std_dev = std_dev[0][0]
        enhanced_image = factor * (image - mean) / std_dev + mean
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image

# Load an image
image_path = r"C:\Users\prash\OneDrive\Desktop\data07\10.jpg"
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print(f"Error: Unable to load the image from {image_path}")
else:
    # Increase contrast using covariance
    contrast_enhanced = increase_contrast_with_covariance(image, factor=3.0)

    # Save the contrast-enhanced image
    output_path = r"C:\Users\prash\OneDrive\Desktop\data07\contrast_enhanced.jpg"
    cv2.imwrite(output_path, contrast_enhanced)

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Contrast Enhanced Image", contrast_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Enhanced image saved to: {output_path}")
