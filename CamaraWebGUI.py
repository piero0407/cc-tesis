#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


def key(event):
    global count, frame, x, y, w, h
    cropped = frame[y:h, x:w]
    '''print("pressed", repr(event.char))'''
    if event.char == '1':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '2':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '3':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '4':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '5':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '6':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '7':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)
    if event.char == '8':
        cv2.imwrite(r".\\SavedFrames\\frame%c_number%d.jpg" %
                    (event.char, count), cropped)


def getContours(thresh):
    # Find contours of the filtered frame
    thresh, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (contours == []):
        cx, cy = int(W*.5), int(H*.5)
        return [], [], cx, cy

    max_area = 100
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area > max_area):
            max_area = area
            ci = i

    # Largest area contour
    cnts = contours[ci]

    # Find convex hull
    hull = cv2.convexHull(cnts)

    # Find convex defects
    hull2 = cv2.convexHull(cnts, returnPoints=False)
    defects = cv2.convexityDefects(cnts, hull2)

    FarDefect = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
    #   cv2.line(correctedFrame, start, end, [0, 255, 0], 1)
    #   cv2.circle(correctedFrame, far, 10, [100, 255, 255], 3)

    # Find moments of the largest contour
    moments = cv2.moments(cnts)

    # Central mass of first order moments
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)

    # Draw center mass
    # cv2.circle(correctedFrame, centerMass, 7, [100, 0, 255], 2)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(correctedFrame, 'Center', tuple(centerMass),
    #            font, 2, (255, 255, 255), 2)

    # Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0, len(FarDefect)):
        x = np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(
            np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
        distanceBetweenDefectsToCenter.append(distance)

    # Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

    # Get fingertip points from contour hull
    # If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0, len(hull) - 1):
        if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 80) or (np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 80):
            if hull[i][0][1] < 500:
                finger.append(hull[i][0])

    # The fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])
    fingers = finger[0:5]

    # Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0, len(fingers)):
        distance = np.sqrt(np.power(
            fingers[i][0] - centerMass[0], 2) + np.power(fingers[i][1] - centerMass[0], 2))
        fingerDistance.append(distance)

    # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    # than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0, len(fingers)):
        if fingerDistance[i] > AverageDefectDistance + 130:
            result = result + 1

    # Print number of pointed fingers
    # cv2.putText(correctedFrame, str(result),
    #            (100, 100), font, 2, (255, 255, 255), 2)

    return cnts, hull, cx, cy


def show_frame():
    global count, frame, cx, cy, x, y, w, h
    '''Getting image from the Camera'''
    ret, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    '''Configure image'''
    correctedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Blur the image
    blur = cv2.blur(frame, (3, 3))

    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=3)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=3)
    filtered = cv2.medianBlur(dilation2, 5)

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation3, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)

    cnts, hull, cx, cy = getContours(filtered)

    # Print bounding rectangle
    # x, y, w, h = cv2.boundingRect(cnts)
    x, y, w, h = cx - int(W * .5), cy - int(H * 0.5), cx + \
        int(W * .5), cy + int(H * 0.5)
    # dw, dh = w / W, h / H
    # w, h = int(w * dh), int(h * dh)

    if x < 0:
        x = 0
    elif y < 0:
        y = 0
    elif w > frame.shape[1]:
        w = frame.shape[1]
    elif h > frame.shape[0]:
        h = frame.shape[0]

    # img = cv2.rectangle(correctedFrame, (x, y),
    #                    (x + w, y + h), (0, 128, 128), 2)
    # cv2.drawContours(correctedFrame, [hull], -1, (255, 255, 255), 2)

    cropped = correctedFrame[y:h, x:w]
    cropped = cv2.resize(cropped, (W, H))

    '''Show Image on the TK Window'''
    img1 = Image.fromarray(correctedFrame)
    imgtk1 = ImageTk.PhotoImage(image=img1)
    lmain.imgtk = imgtk1
    lmain.configure(image=imgtk1)

    img2 = Image.fromarray(cropped)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lcrop.imgtk = imgtk2
    lcrop.configure(image=imgtk2)

    count += 1

    lmain.after(5, show_frame)


videoCapture = cv2.VideoCapture(0)

window = tk.Tk()
window.bind('<Escape>', lambda e: window.quit())
window.bind("<Key>", key)
lmain = tk.Label(window)
lcrop = tk.Label(window)
lmain.pack(side=tk.LEFT)
lcrop.pack(side=tk.LEFT)

count = 0.0
ret, frame = videoCapture.read()
W, H = 288, 352
x, y, w, h = 0, 0, 288, 352

show_frame()
window.mainloop()
videoCapture.release()

'''width, height = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))'''
'''
#!/usr/bin/env python
import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid()

app = Application()
app.master.title('Sample application')
app.mainloop()
'''
