import cv2
import numpy as np

def draw_polygon_around_color(image_path, lower_bound, upper_bound):
    """
    Forms a polygon around a specific grayscale range in the image and draws its boundary.
    
    Parameters:
    - image_path: str, path to the input image.
    - lower_bound: int, lower grayscale threshold.
    - upper_bound: int, upper grayscale threshold.
    
    Returns:
    - result_image: Image with polygons drawn.
    - polygon_params: List of parameters (area, perimeter, vertices) for each polygon.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None

    # Create a binary mask based on the grayscale range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the grayscale image to BGR for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # List to store polygon parameters
    polygon_params = []

    for contour in contours:
        # Skip small contours (noise)
        if cv2.contourArea(contour) < 100:  # Adjust this threshold as needed
            continue

        # Approximate the polygon around the contour
        epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust 0.01 for more/less precision
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the polygon on the image
        cv2.polylines(color_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # Green boundary

        # Calculate and store polygon parameters
        area = cv2.contourArea(polygon)
        perimeter = cv2.arcLength(polygon, True)
        vertices = polygon.reshape(-1, 2).tolist()  # Extract (x, y) coordinates
        polygon_params.append({"area": area, "perimeter": perimeter, "vertices": vertices})

    return color_image, polygon_params


# Image path
image_path = "C:/Users/prash/OneDrive/Desktop/open cv/area_3.PNG"

# Grayscale range to filter (adjust as needed)
lower_gray = 100  # Lower grayscale threshold
upper_gray = 200  # Upper grayscale threshold

# Process the image and extract polygon parameters
result_image, polygons = draw_polygon_around_color(image_path, lower_gray, upper_gray)

# Display the result
if result_image is not None:
    cv2.imshow("Polygon Around Color", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    output_path = "polygon_around_color.png"
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved at {output_path}")

    # Print polygon parameters
    if polygons:
        print("Polygon Parameters:")
        for idx, poly in enumerate(polygons):
            print(f"Polygon {idx + 1}:")
            print(f"  Area: {poly['area']} pixels")
            print(f"  Perimeter: {poly['perimeter']} pixels")
            print(f"  Vertices: {poly['vertices']}")
