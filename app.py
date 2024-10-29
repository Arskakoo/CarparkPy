import cv2
import numpy as np

# Set desired display dimensions
WIDTH, HEIGHT = 1600, 900

def load_and_resize_image(path, target_height):
    img = cv2.imread(path)
    return cv2.resize(img, (int(img.shape[1] * (target_height / img.shape[0])), target_height))

def detect_parking_slots(reference_image_path, target_image_path):
    # Load and resize images
    ref_img = load_and_resize_image(reference_image_path, HEIGHT)
    target_img = load_and_resize_image(target_image_path, HEIGHT)

    # Preprocessing: Convert to grayscale and blur
    target_blurred = cv2.GaussianBlur(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # Edge detection and Hough Transform to detect lines
    edges = cv2.Canny(target_blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a blank image to draw lines on
    line_image = np.zeros_like(target_img)

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:  # Unpack line coordinates
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Combine images
    combined_image = cv2.addWeighted(target_img, 0.8, line_image, 1, 0)
    matched_img = np.hstack((ref_img, combined_image))

    # Resize and display the matched image
    cv2.imshow('Matched Features', cv2.resize(matched_img, (WIDTH, HEIGHT)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_parking_slots('image/Border.jpg', 'image/Example.jpg')
