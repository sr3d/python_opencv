import numpy as np
import cv2
import sys
import time
import pkg_resources


webcam = cv2.VideoCapture(1)

width = 640
height = 480
webcam.set(3, width)
webcam.set(4, height)

windows = [
    # "gray", "edges", "process", "binary", 'visualize',
    "final"
]

for i, window_name in enumerate(windows):
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 20 + width * i, height)


while True:
    # Capture frame-by-frame
    ret, image = webcam.read()
    if not ret:
        break

    mean, std = cv2.meanStdDev(image)
    color = (int(mean[0][0]), int(mean[1][0]), int(mean[2][0]))
    cv2.rectangle(image, (0, 0), (width, height), color, 150)
    cv2.imshow("final", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()