import cv2
import numpy as np

original_img = cv2.imread("lane.jpg")
lane_image = np.copy(original_img)

# Convert lane_image to grayscale
grayscale_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Filter the grayscale_image to reduce noise
blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

cv2.imshow('Lane', blur_image)
cv2.waitKey(0)