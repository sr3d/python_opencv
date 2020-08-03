import cv2
import sys
import time
import pkg_resources



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


webcam = cv2.VideoCapture(0)

width = 640
height = 480
webcam.set(3, width)
webcam.set(4, height)

for i, window_name in enumerate(["binary", "gray", "final"]):
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 20 + width * i, height)


while True:
    # Capture frame-by-frame
    ret, image = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    # https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
    # edges = cv.Canny(img,100,200)

    cv2.imshow("gray", gray)

    _, binary = cv2.threshold(gray,
        150, # threshold
        255, # new value
        cv2.THRESH_BINARY_INV)
    cv2.imshow("binary", binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(contour) for contour in contours]
    for (x, y, w, h) in rects:
        if w < 50 and h < 50: continue
        cv2.rectangle(image, (x,y), (x+h, y+h), (0, 255, 0), 1)

    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imshow("final", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()