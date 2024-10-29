import cv2
from matplotlib import pyplot as plt

image_template = cv2.imread('image/Example.jpg')

height = 700
width = 1080

resized_image = cv2.resize(image_template, (width, height)) 

cv2.imwrite('resized_image.jpg', resized_image)
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0) # waits until any key is pressed