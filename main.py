import cv2
import numpy as np


def find_edges_with_canny(image):
    """
        - Convert lane_image to grayscale
        - Filter the grayscale_image to reduce noise
        - Canny Edge detection

        :param image: Lane image
        :return: Canny edge detected image
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image


def apply_region_of_interest(canny_image):
    """
    Applies a region of interest mask to a Canny edge-detected image.
    :param canny_image: Canny edge-detected image
    :return: Masked Canny image within the specified region of interest
    """
    image_height = canny_image.shape[0]
    region_vertices = np.array([[(200, image_height), (1100, image_height), (550, 250)]])
    region_mask = np.zeros_like(canny_image)
    cv2.fillPoly(region_mask, region_vertices, 255)
    masked_canny_image = cv2.bitwise_and(canny_image, region_mask)
    return masked_canny_image


def draw_hough_lines(image, hough_lines):
    """
        Draws detected Hough lines on a black image.
        :param image: Lane image
        :param hough_lines: Hough lines
        :return: Image with the detected lines drawn on it
    """

    # Create a black image with the same dimensions as the input image
    line_image = np.zeros_like(image)
    if hough_lines is not None:
        for line in hough_lines:
            # Extract the coordinates of the two endpoints of the line
            x1, y1, x2, y2 = line.reshape(4)
            # Draw a blue line on the 'line_image' using the extracted coordinates with a thickness of 10px
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


original_img = cv2.imread("lane.jpg")
lane_image = np.copy(original_img)

# Canny edge detected output
canny_output = find_edges_with_canny(lane_image)

# Mask canny image
isolated_region = apply_region_of_interest(canny_output)

# Using Hough Line Transform to detect lines in an image.
detected_lines = cv2.HoughLinesP(isolated_region, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# Drae lines on the image
line_output_image = draw_hough_lines(lane_image, detected_lines)

# Combine the original lane image with the image containing detected lines
blended_image = cv2.addWeighted(lane_image, 0.8, line_output_image, 1, 1)

# Show result
cv2.imshow('Lane', blended_image)
cv2.waitKey(0)
