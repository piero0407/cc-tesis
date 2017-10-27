import gzip as gz
import os
import pickle as pkl
from glob import iglob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cv2


numPix = 128


def labelToIndex(data):
    global datasetLabels
    for i in range(len(datasetLabels)):
        if datasetLabels[i] == data:
            return i
    return -1 


def labelToVector(data):
    global datasetLabels
    return np.array([1 if (datasetLabels[j] == data) else 0 for j in range(
        len(datasetLabels))])


def marcel():
    global numPix, datasetLabels
    datasetLabels = ['A', 'B', 'C', 'F', 'P', 'V']

    dataset = [[], []]
    labels = [[], []]

    # Cargar todos las imagenes de una carpeta y sus sub carpetas
    tePath = Path(r"./Marcel-data")
    teList = [f for f in tePath.resolve().glob('**/*') if f.is_file()]

    for f in teList:
        fileName = str(f)
        if fileName.find("Marcel-Train") != -1:
            labelIndex = 0
        else:
            labelIndex = 1
        # Imagen a escala de grises
        image = cv2.imread(fileName, flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(numPix, numPix)).reshape(
            (1, numPix * numPix))  # Imagen en matris (100x100) a vector (1x10000)
        dataset[labelIndex].append(image)  # Guardar imagen en arreglo de imagenes
        labels[labelIndex].append(labelToVector(f.stem[0]))  # Guardar etiqueta en arreglo de etiquetas
    
    x_trn = np.array(dataset[0]).reshape(len(dataset[0]), numPix * numPix)
    y_trn = np.array(labels[0]).reshape(len(labels[0]), len(datasetLabels))
    x_val = np.array(dataset[1]).reshape(len(dataset[1]), numPix * numPix)
    y_val = np.array(labels[1]).reshape(len(labels[1]), len(datasetLabels))

    # Guardar los arreglos en un archivo pkl.gz
    dsguardar = [(np.array(x_trn), np.array(y_trn)), (np.array(x_val), np.array(y_val))]

    f = gz.open('marcel.pkl.gz', 'wb')
    pkl.dump(dsguardar, f)
    f.close()


def dataset():
    global numPix, datasetLabels
    datasetLabels = ['1', '2', '3', '4', '5', '6']

    dataset = [[], [], [], [], [], []]
    labels = [[], [], [], [], [], []]

    # Cargar todos las imagenes de una carpeta y sus sub carpetas
    tePath = Path(r"./Dataset Final")
    # tePath = Path(r"./Dataset Final")
    teList = [f for f in tePath.resolve().glob('**/*') if f.is_file()]

    # Recorrer todas las imagenes
    for f in teList:
        fileName = str(f)
        labelIndex = labelToIndex(f.stem[0])
        # Imagen a escala de grises
        image = cv2.imread(fileName, flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(numPix, numPix)).reshape(
            (1, numPix * numPix))  # Imagen en matris (100x100) a vector (1x10000)
        dataset[labelIndex].append(image)  # Guardar imagen en arreglo de imagenes
        labels[labelIndex].append(labelToVector(f.stem[0]))  # Guardar etiqueta en arreglo de etiquetas

    # plt.imshow(dataset[25].reshape(numPix, numPix), cmap='gray')
    # plt.show()
    # print(labels[25])

    x_trn, y_trn, x_val, y_val = [], [], [], []

    for i in range(len(dataset)):
        # 90% de las imagenes son para entrenamiento y 10% para validación
        m_trni = int(len(dataset[i]) * .8)
        x_trni = np.array(dataset[i][:m_trni]).reshape(m_trni, numPix * numPix)
        y_trni = np.array(labels[i][:m_trni]).reshape(m_trni, len(datasetLabels))

        m_vali = int(len(dataset[i]) * .2)
        x_vali = np.array(dataset[i][m_trni:m_trni + m_vali]).reshape(m_vali, numPix * numPix)
        y_vali = np.array(labels[i][m_trni:m_trni + m_vali]).reshape(m_vali, len(datasetLabels))

        x_trn.extend(x_trni)
        y_trn.extend(y_trni)
        x_val.extend(x_vali)
        y_val.extend(y_vali)

    # Guardar los arreglos en un archivo pkl.gz
    dsguardar = [(np.array(x_trn), np.array(y_trn)), (np.array(x_val), np.array(y_val))]

    f = gz.open('dataset.pkl.gz', 'wb')
    pkl.dump(dsguardar, f)
    f.close()

if __name__ == '__main__':
    marcel()
