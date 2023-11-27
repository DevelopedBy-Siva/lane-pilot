import cv2
import numpy as np


def find_edges_with_canny(image):
    """
        1. Convert lane_image to grayscale
        2. Filter the grayscale_image to reduce noise
        3. Canny Edge detection

        :param image: Lane image
        :return: Canny edge detected image
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image


original_img = cv2.imread("lane.jpg")
lane_image = np.copy(original_img)

# Canny edge detected output
canny_output = find_edges_with_canny(lane_image)

# Show result
cv2.imshow('Lane', canny_output)
cv2.waitKey(0)
