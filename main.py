import cv2
import numpy as np

from image_processing import find_edges_with_canny, apply_region_of_interest, draw_hough_lines, \
    calculate_average_coordinates


def main():
    lane_video = cv2.VideoCapture("resources/video.mp4")
    while lane_video.isOpened():
        _, frame = lane_video.read()
        if frame is None:
            break
        canny_output = find_edges_with_canny(frame)
        isolated_region = apply_region_of_interest(canny_output)
        detected_lines = cv2.HoughLinesP(isolated_region, 2, np.pi / 180, 100, np.array([]), minLineLength=40,
                                         maxLineGap=5)
        averaged_lane_coordinates = calculate_average_coordinates(frame, detected_lines)
        line_output_image = draw_hough_lines(frame, averaged_lane_coordinates)
        blended_image = cv2.addWeighted(frame, 0.8, line_output_image, 1, 1)
        cv2.imshow('Lane Detection', blended_image)
        if cv2.waitKey(1) == ord('q'):
            break
    lane_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
