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


def region_of_interest(image):
    """
    Mask the image to separate the region
    :param image: Canny image
    :return: masked canny image
    """
    image_height = image.shape[0]
    views = np.array([[(200, image_height), (1100, image_height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, views, 255)
    masked_canny = cv2.bitwise_and(image, mask)
    return masked_canny


def display_lines(image, line):
    return ""


original_img = cv2.imread("lane.jpg")
lane_image = np.copy(original_img)

# Canny edge detected output
canny_output = find_edges_with_canny(lane_image)

# Mask canny image
isolated_region = region_of_interest(canny_output)

# Using Hough Line Transform to detect lines in an image.
lines = cv2.HoughLinesP(isolated_region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_image, lines)

# Show result
cv2.imshow('Lane', isolated_region)
cv2.waitKey(0)
