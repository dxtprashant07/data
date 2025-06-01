import cv2
import numpy as np

# Global variables
points = []

# Conversion factor (e.g., meters per pixel)
# Replace this value based on the satellite image resolution
meters_per_pixel = 0.5  # Example: 0.5 meters per pixel

# Mouse callback function to select points
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Select Field", image)

def calculate_area(points):
    contour = np.array(points, dtype=np.int32)
    pixel_area = cv2.contourArea(contour)
    real_area = pixel_area * (meters_per_pixel ** 2)
    return real_area, pixel_area

# Load the satellite image
image = cv2.imread("field_image.jpg")  # Replace with your satellite image path
clone = image.copy()

cv2.imshow("Select Field", image)
cv2.setMouseCallback("Select Field", select_point)

print("Click to select boundary points. Press 'c' to calculate area and 'r' to reset.")

while True:
    key = cv2.waitKey(1) & 0xFF
    
    # Calculate area
    if key == ord("c"):
        if len(points) > 2:
            real_area, pixel_area = calculate_area(points)
            print(f"Selected Area: {real_area:.2f} m² (Pixel Area: {pixel_area:.2f} px²)")
            cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Select Field", image)
        else:
            print("At least 3 points are needed to calculate area.")
    
    # Reset the selection
    elif key == ord("r"):
        points.clear()
        image = clone.copy()
        cv2.imshow("Select Field", image)
        print("Selection reset.")
    
    # Exit the program
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
