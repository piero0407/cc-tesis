import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk


hand_cascade = cv2.CascadeClassifier('Camara Gestos/aGest.xml')
images = load_images_from_folder("NewFrames", mirror)

counter = 0
index = ''
for image in images:
    frame = image[0]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hand_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('img', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    basename = os.path.splitext(image[1])[0]
    newName = "G" + basename[5]

    cv2.imwrite(r"NewFrames/" + newName + "_DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = frame.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 45, 1)
    frame = cv2.warpAffine(frame, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_45DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_45IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_45UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_45UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = frame.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 45, 1)
    frame = cv2.warpAffine(frame, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_90DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_90IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_90UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_90UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = frame.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 45, 1)
    frame = cv2.warpAffine(frame, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_135DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_135IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_135UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_135UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    counter = counter + 1
    '''