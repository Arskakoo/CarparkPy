import cv2
import numpy as np


image = cv2.imread('resized_image.jpg')  
image_height, image_width = image.shape[:2]  

box_size = 15

def draw_box(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        top_left = (x, y)
        bottom_right = (x + box_size, y + box_size)

        if bottom_right[0] <= image_width and bottom_right[1] <= image_height:
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)  
            cv2.imshow('Park', image)

cv2.namedWindow('Park')
cv2.setMouseCallback('Park', draw_box)

while True:
    cv2.imshow('Park', image)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cv2.destroyAllWindows()
6