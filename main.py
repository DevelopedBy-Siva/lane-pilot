import cv2

img = cv2.imread("lane.jpg")
cv2.imshow('Lane', img)
cv2.waitKey(0)