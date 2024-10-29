import cv2
import numpy as np

# Load the image
image_template = cv2.imread('image/Example.jpg')

# Resize the image
width = 1600
height = 900
resized_image = cv2.resize(image_template, (width, height))

# Convert to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and filter for parking slots
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # Filter based on area (adjust these values as needed)
    if area > 500 or area > 5000:  # Example area limits
        continue

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour has 4 points (indicating a rectangle)
    if len(approx) == 4:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h
        
        # Filter based on aspect ratio (adjust these values as needed)
        if 2 < aspect_ratio < 5:  # Example aspect ratio limits
            # Draw the detected parking slot
            cv2.drawContours(resized_image, [approx], -1, (0, 255, 0), 2)

# Show the result
cv2.imshow('Detected Parking Slots', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
