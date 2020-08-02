import cv2
import sys
import time
import pkg_resources



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


webcam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, image = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30)
     )

    # eyes = eye_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor = 1.2,
    #     minNeighbors = 5,
    #     minSize = (30,30)
    #  )


    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x,y),
            (x+h, y+h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Faces found", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
