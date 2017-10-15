import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk

import os
from pathlib import Path
from glob import iglob

tePath = Path("./SavedFrames2")
teList = [f for f in tePath.resolve().glob('**/*') if f.is_file()]

counter = 0
index = ''
for f in teList:
    image = cv2.imread(str(f))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('img', gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    basename = os.path.splitext(f.name)[0]
    newName = basename[:2]

    frame = image
    cv2.imwrite(r"NewFrames/" + newName + "_DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = image.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 45, 1)
    frame = cv2.warpAffine(image, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_45DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_45IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_45UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_45UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = image.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 90, 1)
    frame = cv2.warpAffine(image, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_90DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_90IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_90UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_90UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    frameShape = image.shape
    ret = cv2.getRotationMatrix2D(
        (frameShape[1] / 2.0, frameShape[0] / 2.0), 135, 1)
    frame = cv2.warpAffine(image, ret, ((frameShape[1], (frameShape[0]))))
    cv2.imwrite(r"NewFrames/" + newName + "_135DER" +
                str(counter) + ".jpg", frame)
    cv2.imwrite(r"NewFrames/" + newName + "_135IZQ" +
                str(counter) + ".jpg", cv2.flip(frame, 1))
    cv2.imwrite(r"NewFrames/" + newName + "_135UPDER" +
                str(counter) + ".jpg", cv2.flip(frame, 0))
    cv2.imwrite(r"NewFrames/" + newName + "_135UPIZQ" +
                str(counter) + ".jpg", cv2.flip(frame, -1))

    counter = counter + 1