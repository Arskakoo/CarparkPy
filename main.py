import cv2
import numpy as np
from matplotlib import pyplot as plt

# Image resize
image_template = cv2.imread('image/Example.jpg')

height = 900
width = 1600

# Resize the image
resized_image = cv2.resize(image_template, (width, height)) 
cv2.imwrite('resized_image.jpg', resized_image)

# Parkslots draw
image = cv2.imread('resized_image.jpg')

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for white color
lower_white = np.array([0, 0, 200])  
upper_white = np.array([180, 25, 255])  
mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Change masked white areas to red
image[mask > 0] = [0, 0, 255]

# Find contours of the masked white areas
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected contours
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Only draw rectangles for significant areas
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw in blue

# Show the modified image with rectangles
cv2.imshow("Parkkipaikka ruutuineen", image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
