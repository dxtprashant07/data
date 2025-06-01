import cv2
import numpy as np
import json
import os

def draw_polygon_with_geojson_structure(image_path, lower_bound, upper_bound, corners, output_dir, epsilon_factor=0.001, contour_area_threshold=10):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    # Image dimensions
    height, width = image.shape

    # Unpack the corner coordinates
    top_left, top_right, bottom_right, bottom_left = corners

    # Map geographic coordinates to pixel coordinates
    def geo_to_pixel(coord):
        x_geo, y_geo = coord
        x_pixel = int(((x_geo - top_left[0]) / (top_right[0] - top_left[0])) * width)
        y_pixel = int(((y_geo - top_left[1]) / (bottom_left[1] - top_left[1])) * height)
        return x_pixel, y_pixel

    # Map pixel coordinates back to geographic coordinates
    def pixel_to_geo(pixel):
        x_pixel, y_pixel = pixel
        x_geo = top_left[0] + (x_pixel / width) * (top_right[0] - top_left[0])
        y_geo = top_left[1] + (y_pixel / height) * (bottom_left[1] - top_left[1])
        return float(x_geo), float(y_geo)

    # Apply GaussianBlur for noise reduction
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Create a binary mask based on the grayscale range
    mask = cv2.inRange(blurred_image, lower_bound, upper_bound)

    # Optional: Apply Adaptive Thresholding (useful for images with varying lighting)
    # mask = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the grayscale image to BGR for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Process each contour and save as GeoJSON
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        
        # Skip small contours (noise)
        if contour_area < contour_area_threshold:
            continue

        # Approximate the polygon around the contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Convert polygon vertices to floating-point geographic coordinates
        vertices = [pixel_to_geo(tuple(pt[0])) for pt in polygon]

        # Ensure the polygon is closed
        if len(vertices) > 0 and not np.allclose(vertices[0], vertices[-1]):
            vertices.append(vertices[0])  # Add the first vertex at the end to close the polygon

        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"area": contour_area},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [vertices]
                    }
                }
            ]
        }

        # Save the GeoJSON to a file
        geojson_filename = os.path.join(output_dir, f"polygon_{i + 1}.json")
        with open(geojson_filename, 'w') as json_file:
            json.dump(geojson, json_file, indent=4)
        print(f"Polygon {i + 1} saved as GeoJSON in {geojson_filename}")

        # Draw the polygon on the image
        cv2.polylines(color_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # Green boundary

    return color_image


# Main Code
if __name__ == "__main__":
    # Path to the input image
    image_path = "C:/Users/prash/OneDrive/Desktop/class_mask/agriculture/agriculture.png"

    # Grayscale range to filter (adjust as needed)
    lower_gray = 100  # Lower grayscale threshold
    upper_gray = 200  # Upper grayscale threshold

    # User-provided geographic bounding box corners (float values)
    user_top_left = (77.64210216899312, 29.519690553168758)
    user_top_right = (77.79366733198111, 29.519690553168758)
    user_bottom_right = (77.79366733198111, 29.408070020352426)
    user_bottom_left = (77.64210216899312, 29.408070020352426)

    # List of corners in clockwise order
    user_corners = [user_top_left, user_top_right, user_bottom_right, user_bottom_left]

    # Create an output directory to save GeoJSON files
    output_dir = "masking_output"

    # Process the image and save polygons as GeoJSON
    result_image = draw_polygon_with_geojson_structure(
        image_path, lower_gray, upper_gray, user_corners, output_dir, epsilon_factor=0.001, contour_area_threshold=10
    )

    # Display the result
    if result_image is not None:
        cv2.imshow("Polygons as GeoJSON", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Save the result image
        output_image_path = os.path.join(output_dir, "polygon_geojson.png")
        cv2.imwrite(output_image_path, result_image)
        print(f"Result image saved at {output_image_path}")
