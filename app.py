import cv2
import numpy as np

reference_image_path = 'image/Example.jpg'
target_image_path = 'image/Example.jpg'

def detect_parking_slots(reference_image_path, target_image_path):
    # Load images
    ref_img = cv2.imread(reference_image_path)
    target_img = cv2.imread(target_image_path)

    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Detect edges in the reference image
    ref_edges = cv2.Canny(ref_gray, 50, 150, apertureSize=3)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_edges, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target_gray, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_descriptors, target_descriptors)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    matched_img = cv2.drawMatches(ref_img, ref_keypoints, target_img, target_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # (Optional) Calculate bounding boxes around detected parking spots
    # Here you would typically apply some logic to determine the locations of the parking spots based on matched keypoints.
    # This part will depend on the layout and expected patterns of the parking slots.

    # Display images
    cv2.imshow('Matched Features', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_parking_slots('reference_image.jpg', 'target_image.jpg')
