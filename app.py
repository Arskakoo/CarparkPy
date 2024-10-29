import cv2
import numpy as np

# Set desired display dimensions
width = 1600
height = 900

def detect_parking_slots(reference_image_path, target_image_path):
    # Load images
    ref_img = cv2.imread(reference_image_path)
    target_img = cv2.imread(target_image_path)

    # Resize the target image to match the reference image height
    target_img_resized = cv2.resize(target_img, (int(ref_img.shape[1] * (ref_img.shape[0] / target_img.shape[0])), ref_img.shape[0]))

    # Preprocessing: Convert to grayscale and blur
    target_gray = cv2.cvtColor(target_img_resized, cv2.COLOR_BGR2GRAY)
    target_blurred = cv2.GaussianBlur(target_gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(target_blurred, 50, 150, apertureSize=3)

    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the target image to draw lines on
    line_image = np.zeros_like(target_img_resized)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw lines with thickness of 3

    # Combine the original target image with the line image
    combined_image = cv2.addWeighted(target_img_resized, 0.8, line_image, 1, 0)

    # Create a blank image to draw matches
    matched_img = np.hstack((ref_img, combined_image))  # Combine images side by side

    # Resize the matched image to the specified dimensions
    matched_img_resized = cv2.resize(matched_img, (width, height))

    # Display images
    cv2.imshow('Matched Features', matched_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_parking_slots('image/Border.jpg', 'image/Example.jpg')
