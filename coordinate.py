

import cv2
import numpy as np

image_path = 'your name.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()


r = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)


x_start, y_start, crop_width, crop_height = [int(v) for v in r]


if crop_width > 0 and crop_height > 0:
    
    cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

    
    cropped_y_coords, cropped_x_coords = np.mgrid[y_start:y_start + crop_height, x_start:x_start + crop_width]
    cropped_pixel_coordinates = np.column_stack((cropped_x_coords.ravel(), cropped_y_coords.ravel()))

    
    print(f"Total pixel coordinates stored for the cropped section: {len(cropped_pixel_coordinates)}")

    
    print("First few pixel coordinates of the cropped section:")
    print(cropped_pixel_coordinates) 

    
    cv2.imshow("Cropped Image", cropped_image)
else:
    print("No valid ROI selected. Please try again.")

cv2.waitKey(0)
cv2.destroyAllWindows()