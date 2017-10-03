import numpy as np
import cv2
import pickle as pkl
import gzip as gz
import matplotlib.pyplot as plt

import os
from pathlib import Path
from glob import iglob

numPix = 100
dataset = []
labels = []

# videoCapture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('Camara Gestos/xml/haarcascade_frontalface_default.xml') 

tePath = Path("Camara Gestos/images")
tePath = Path("SavedFrames")
images = [f for f in tePath.resolve().glob('**/*') if f.is_file()]

# ret, image = videoCapture.read()

filePath = images[1]
fileName = str(filePath)
image = cv2.imread(fileName)

faces = face_cascade.detectMultiScale(image, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y+h), (255, 0, 0), cv2.FILLED)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
mask2 = cv2.inRange(ycc, np.array([54, 131, 110]), np.array([163, 157, 135]))

kernel_square = np.ones((11, 11), np.uint8)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

edges = cv2.Canny(mask2, 200, 200)
dilation = cv2.dilate(mask2, kernel_ellipse, iterations=3)
erosion = cv2.erode(dilation, kernel_square, iterations=2)
dilation = cv2.dilate(erosion, kernel_ellipse, iterations=3)
'''
for i in range(2):
    erosion = cv2.erode(dilation, kernel_square, iterations=2)
    dilation = cv2.dilate(erosion, kernel_ellipse, iterations=3)
'''
filtered = cv2.medianBlur(dilation, 5)
ret, thresh = cv2.threshold(filtered, 127, 255, 0)

res = cv2.bitwise_and(image, image, mask=thresh)

plt.imshow(ycc, cmap='gray')
plt.show()
plt.imshow(thresh, cmap='gray')
plt.show()
# videoCapture.release()

'''
image = cv2.resize(image, dsize=(numPix, numPix)).reshape((1, numPix * numPix))  # Imagen en matris (100x100) a vector (1x10000)
dataset.append(image)  # Guardar imagen en arreglo de imagenes
labels.append(filePath.stem[0])  # Guardar etiqueta en arreglo de etiquetas

plt.imshow(dataset[0].reshape(numPix, numPix), cmap='gray')
plt.show()
print(labels[0])
'''