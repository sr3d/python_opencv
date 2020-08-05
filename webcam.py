import numpy as np
import cv2
import sys
import time
import pkg_resources

from sklearn.cluster import KMeans


# import AppKit
# [(screen.frame().size.width, screen.frame().size.height)
#     for screen in AppKit.NSScreen.screens()]


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


webcam = cv2.VideoCapture(1)

width = 640
height = 480
webcam.set(3, width)
webcam.set(4, height)

for i, window_name in enumerate([ "gray", "edges", "process", "binary", 'visualize', "final"]):
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 20 + width * i, height)


def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect


kernel = np.ones((5,5),np.uint8)

while True:
    # Capture frame-by-frame
    ret, image = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    cv2.imshow("gray", gray)

    # https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
    edges = cv2.Canny(image,100,200)
    cv2.imshow("edges", edges)


    # edges = cv2.dilate(edges, kernel, iterations=6)
    # # dilation = cv2.dilate(img,kernel,iterations = 1)
    # cv2.imshow("process", edges)


    _, binary = cv2.threshold(gray,
        150, # threshold
        255, # new value
        cv2.THRESH_BINARY_INV)
    cv2.imshow("binary", binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # only grab big enough contours
    # only find rectangle contours
    # rects = [cv2.boundingRect(contour) for contour in contours]
    # approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    rectangle_contours = []

    # cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
    for contour in contours:
        # len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt)
        approx = cv2.approxPolyDP(
            contour,
            0.02 * cv2.arcLength(contour, True),
            True)
        if len(approx) == 4: #  and cv2.isContourConvex(contour):
            rectangle_contours.append(contour)

    rects = [cv2.boundingRect(contour) for contour in rectangle_contours]
    # rects = [cv2.boundingRect(contour) for contour in contours]
    for (x, y, w, h) in rects:
        if w < 50 and h < 50: continue
        cv2.rectangle(image, (x,y), (x+h, y+h), (0, 255, 0), 1)


    # # Find and display most dominant colors
    # reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster = KMeans(n_clusters=5).fit(reshape)
    # visualize = visualize_colors(cluster, cluster.cluster_centers_)
    # visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
    # cv2.imshow('visualize', visualize)


    image = cv2.drawContours(image, rectangle_contours, -1, (0, 255, 0), 2)
    cv2.imshow("final", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()