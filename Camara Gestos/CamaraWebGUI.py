#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

# print(cv2.getBuildInformation())
face_cascade = cv2.CascadeClassifier('Camara Gestos/xml/haarcascade_frontalface_default.xml')


def key(event):
    global count, cropped
    if event.char in ['1', '2', '3', '4', '5', '6']:
        cv2.imwrite(r"SavedFrames2/G%c_number%d.jpg" %
                    (event.char, count), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    print(event)


def up(event):
    global state
    state += 1
    if(state > 7):
        state = 7


def down(event):
    global state
    state -= 1
    if(state < 1):
        state = 1


def getContours(thresh):
    # Find contours of the filtered frame
    thresh, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (contours == []):
        cx, cy = int(W*.5), int(H*.5)
        return [], [], cx, cy

    min_area = 100
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area < min_area):
            cv2.fillPoly(thresh, cnt, (0, 0, 0))

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


def drawLines(image, points, color, thicc):
    for i in range(len(points)):
        start = tuple(points[i])
        if(i != len(points)):
            end = tuple(points[i+1])
        else:
            end = tuple(points[0])
        cv2.line(image, start, end, color, thicc)


def show_frame():
    global count, fgbgy, fgbgcr, fgbgcb, cropped
    '''Getting image from the Camera'''
    ret, image = videoCapture.read()
    image = cv2.flip(image, 1)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    y = fgbgy.apply(ycc[:, :, 0], ret, .0001)
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
    cr = fgbgcr.apply(ycc[:, :, 1], ret, .0001)
    cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
    cb = fgbgcb.apply(ycc[:, :, 2], ret, .0001)
    cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)

    fgmask = cv2.addWeighted(y, 1, cr, 1, 1)
    fgmask = cv2.addWeighted(fgmask, 1, cb, 1, 1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    ret, fgmask = cv2.threshold(fgmask, 127, 255, 0)
    '''
    edges = cv2.Canny(ycc, 100, 100)
    ret, edges = cv2.threshold(edges, 127, 255, 0)
    img2 = np.zeros_like(image)
    img2[:, :, 0] = edges
    img2[:, :, 1] = edges
    img2[:, :, 2] = edges
    foregroundImage = cv2.addWeighted(ycc, 1, img2, 1, 1)
    '''
    foregroundImage = cv2.bitwise_and(ycc, ycc, mask=fgmask)
    foregroundImageShow = cv2.bitwise_and(image, image, mask=fgmask)

    for (x, y, w, h) in faces:
        cv2.rectangle(foregroundImage, (x, y), (x+w, y+h), (0, 0, 0), cv2.FILLED)

    mask2 = cv2.inRange(foregroundImage, np.array([0, 131, 110]), np.array([255, 157, 135]))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.medianBlur(mask2, 5)
    ret, thresh = cv2.threshold(mask2, 127, 255, 0)

    thresh, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    if (contours != []):        
        min_area = image.shape[0]*image.shape[1]*.01
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area < min_area):
                cv2.fillPoly(thresh, cnt, (0, 0, 0))
            else:
                epsilon = 0.01*cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                #  drawLines(image, approx, (0, 255, 0), 1)
                # cv2.fillPoly(thresh, cnt, (255, 255, 255))
    '''
    ret, thresh = cv2.threshold(thresh, 127, 255, 0)
    res = cv2.bitwise_and(image, image, mask=thresh)

    cnts, hull, cx, cy = getContours(thresh)

    x, y, w, h = cx - int(W * .5), cy - int(H * 0.5), cx + \
        int(W * .5), cy + int(H * 0.5)

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if w > frame.shape[1]:
        w = frame.shape[1]
    if h > frame.shape[0]:
        h = frame.shape[0]
    
    cropped = res[y:h, x:w]
    cropped = cv2.resize(cropped, (W, H))

    '''Show Image on the TK Window'''
    if(state == 1):
        img1 = Image.fromarray(image)
    if(state == 2):
        img1 = Image.fromarray(ycc)
    if(state == 3):
        img1 = Image.fromarray(fgmask)
    if(state == 4):
        img1 = Image.fromarray(foregroundImage)
    if(state == 5):
        img1 = Image.fromarray(mask2)
    if(state == 6):
        img1 = Image.fromarray(thresh)
    if(state == 7):
        img1 = Image.fromarray(res)
    imgtk1 = ImageTk.PhotoImage(image=img1)
    lmain.imgtk = imgtk1
    lmain.configure(image=imgtk1)
    '''
    img2 = Image.fromarray(cropped)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lcrop.imgtk = imgtk2
    lcrop.configure(image=imgtk2)
    '''
    count += 1

    lmain.after(5, show_frame)


videoCapture = cv2.VideoCapture(0)
fgbgy = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
fgbgcr = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
fgbgcb = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
# createBackgroundSubtractorMOG()

window = tk.Tk()
window.bind('<Escape>', lambda e: window.quit())
window.bind("<Key>", key)
window.bind("<Up>", up)
window.bind("<Down>", down)
lmain = tk.Label(window)
#lcrop = tk.Label(window)
lmain.pack(side=tk.LEFT)
#lcrop.pack(side=tk.LEFT)

count = 0.0
ret, frame = videoCapture.read()
state = 1
cropped = np.array([0])
W, H = 288, 352
x, y, w, h = 0, 0, W, H

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
