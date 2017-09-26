import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk


def load_images_from_folder(folder, mirror):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        extension = filename[-6:-4]
        if img is not None:
            if extension != "_m" or (mirror and extension == "_m"):
                images.append((img, filename))
    return images

mirror = False
fgbg = cv2.createBackgroundSubtractorMOG2()
images = load_images_from_folder("NewFrames", mirror)

counter = 0
index = ''
for image in images:
    # image = images[1]
    frame = image[0]
    fgmask = fgbg.apply(frame)

    # Blur the image
    blur = cv2.blur(frame, (3, 3))

    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

    # Kernel matrices for morphological transformation
    kernel_square = np.ones((12, 12), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=3)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=3)
    filtered = cv2.medianBlur(dilation2, 5)

    # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    # dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation3, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)

    basename = os.path.splitext(image[1])[0]
    newName = "G"+basename[5]
    '''if index != basename[5]:
        index = basename[5]
        counter = 0
    '''
    if not mirror:
        cv2.imwrite(r"ProcessedFrames/"+basename+".jpg", thresh)
    else:
        cv2.imwrite(r"NewFrames/"+newName+"_DER"+str(counter)+".jpg", frame)
        cv2.imwrite(r"NewFrames/"+newName+"_IZQ"+str(counter)+".jpg", cv2.flip(frame, 1))
        cv2.imwrite(r"NewFrames/"+newName+"_UPDER"+str(counter)+".jpg", cv2.flip(frame, 0))
        cv2.imwrite(r"NewFrames/"+newName+"_UPIZQ"+str(counter)+".jpg", cv2.flip(frame, -1))
        
        frameShape = frame.shape
        ret = cv2.getRotationMatrix2D((frameShape[1]/2.0, frameShape[0]/2.0), 90, 1)
        frame = cv2.warpAffine(frame,ret,((frameShape[1],(frameShape[0]))))
        cv2.imwrite(r"NewFrames/"+newName+"_90DER"+str(counter)+".jpg", frame)
        cv2.imwrite(r"NewFrames/"+newName+"_90IZQ"+str(counter)+".jpg", cv2.flip(frame, 1))
        cv2.imwrite(r"NewFrames/"+newName+"_90UPDER"+str(counter)+".jpg", cv2.flip(frame, 0))
        cv2.imwrite(r"NewFrames/"+newName+"_90UPIZQ"+str(counter)+".jpg", cv2.flip(frame, -1))

        frameShape = frame.shape
        ret = cv2.getRotationMatrix2D((frameShape[1]/2.0, frameShape[0]/2.0), 180, 1)
        frame = cv2.warpAffine(frame,ret,((frameShape[1],(frameShape[0]))))
        cv2.imwrite(r"NewFrames/"+newName+"_180DER"+str(counter)+".jpg", frame)
        cv2.imwrite(r"NewFrames/"+newName+"_180IZQ"+str(counter)+".jpg", cv2.flip(frame, 1))
        cv2.imwrite(r"NewFrames/"+newName+"_180UPDER"+str(counter)+".jpg", cv2.flip(frame, 0))
        cv2.imwrite(r"NewFrames/"+newName+"_180UPIZQ"+str(counter)+".jpg", cv2.flip(frame, -1))
        
        frameShape = frame.shape
        ret = cv2.getRotationMatrix2D((frameShape[1]/2.0, frameShape[0]/2.0), 270, 1)
        frame = cv2.warpAffine(frame,ret,((frameShape[1],(frameShape[0]))))
        cv2.imwrite(r"NewFrames/"+newName+"_270DER.jpg", frame)
        cv2.imwrite(r"NewFrames/"+newName+"_270IZQ.jpg", cv2.flip(frame, 1))
        cv2.imwrite(r"NewFrames/"+newName+"_270UPDER.jpg", cv2.flip(frame, 0))
        cv2.imwrite(r"NewFrames/"+newName+"_270UPIZQ.jpg", cv2.flip(frame, -1))
        
    counter = counter+1
'''
    cv2.imshow('dst_rt', frame)
    cv2.waitKey(0)
    cv2.imshow('dst_rt', hsv)
    cv2.waitKey(0)
    cv2.imshow('dst_rt', mask2)
    cv2.waitKey(0)
    cv2.imshow('dst_rt', filtered)
    cv2.waitKey(0)
    cv2.imshow('dst_rt', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''