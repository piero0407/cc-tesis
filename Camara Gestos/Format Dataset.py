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

# Cargar todos las imagenes de una carpeta y sus sub carpetas
tePath = Path("Marcel-data")
teList = [f for f in tePath.resolve().glob('**/*') if f.is_file()]

# Recorrer todas las imagenes
for f in teList:
    fileName = str(f)
    image = cv2.imread(fileName, flags=cv2.IMREAD_GRAYSCALE)  # Imagen a escala de grises
    image = cv2.resize(image, dsize=(numPix, numPix)).reshape((1, numPix * numPix))  # Imagen en matris (100x100) a vector (1x10000)
    dataset.append(image)  # Guardar imagen en arreglo de imagenes
    labels.append(f.stem[0])  # Guardar etiqueta en arreglo de etiquetas 

# plt.imshow(dataset[25].reshape(numPix, numPix), cmap='gray')
# plt.show()
# print(labels[25])

# 90% de las imagenes son para entrenamiento y 10% para validación
m_trn = int(len(dataset)*.9)
x_trn = np.array(dataset[:m_trn]).reshape(m_trn, numPix*numPix)
y_trn = np.array(labels[:m_trn]).reshape(m_trn, 1)

m_val = int(len(dataset)*.1)
x_val = np.array(dataset[m_trn:m_trn+m_val]).reshape(m_val, numPix*numPix)
y_val = np.array(labels[m_trn:m_trn+m_val]).reshape(m_val, 1)

# Guardar los arreglos en un archivo pkl.gz
dsguardar = [(x_trn, y_trn), (x_val, y_val)]

f = gz.open('marcel-dataset.pkl.gz', 'wb')
pkl.dump(dsguardar, f)
f.close()