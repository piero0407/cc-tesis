# Getting the MNIST data provided by Tensorflow
import cv2
import numpy as np
import gzip
import matplotlib.pyplot as plt

import os
from pathlib import Path
from glob import iglob

'''
from tensorflow.examples.tutorials.mnist import input_data
# Loading in the mnist data
mnist = input_data.read_data_sets("Marcel-data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.trY, mnist.test.images,\
    mnist.test.trY

print(trX)
'''


def convert_image(imagefilename, scale, w, h):

    img = cv2.imread(imagefilename, flags=cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (w, h))
    resized = resized.reshape((1, -1))
    return resized


scale, w, h = 1, 66, 76

trI = 0
trX = np.zeros((1, w * h))
trY = np.zeros((1, 6))
trPath = Path("Marcel-data/Marcel-Train")
trList = [f for f in trPath.resolve().glob('**/*') if f.is_file()]
for f in trList:
    trI = trI + 1
    vector = convert_image(str(f), scale, w, h)
    trX = np.concatenate((trX, vector))

    s = str(f)[65:].split('\\')
    if s[0] == 'A':
        trY = np.concatenate((trY, np.array([[1, 0, 0, 0, 0, 0]])))
    if s[0] == 'B':
        trY = np.concatenate((trY, np.array([[0, 1, 0, 0, 0, 0]])))
    if s[0] == 'C':
        trY = np.concatenate((trY, np.array([[0, 0, 1, 0, 0, 0]])))
    if s[0] == 'Five':
        trY = np.concatenate((trY, np.array([[0, 0, 0, 1, 0, 0]])))
    if s[0] == 'Point':
        trY = np.concatenate((trY, np.array([[0, 0, 0, 0, 1, 0]])))
    if s[0] == 'V':
        trY = np.concatenate((trY, np.array([[0, 0, 0, 0, 0, 1]])))

    print('Loading train:', trI / len(trList) * 100)
np.delete(trX, 0, 0)
np.delete(trY, 0, 0)

teI = 0
teX = np.zeros((1, w * h))
teY = np.zeros((1, 6))
tePath = Path("Marcel-data/Marcel-Test")
teList = [f for f in tePath.resolve().glob('**/*') if f.is_file()]
for f in teList:
    teI = teI + 1
    vector = convert_image(str(f), scale, w, h)
    teX = np.concatenate((teX, vector))

    s = str(f)[64:].split('\\')
    if s[0] == 'A':
        teY = np.concatenate((teY, np.array([[1, 0, 0, 0, 0, 0]])))
    if s[0] == 'B':
        teY = np.concatenate((teY, np.array([[0, 1, 0, 0, 0, 0]])))
    if s[0] == 'C':
        teY = np.concatenate((teY, np.array([[0, 0, 1, 0, 0, 0]])))
    if s[0] == 'Five':
        teY = np.concatenate((teY, np.array([[0, 0, 0, 1, 0, 0]])))
    if s[0] == 'Point':
        teY = np.concatenate((teY, np.array([[0, 0, 0, 0, 1, 0]])))
    if s[0] == 'V':
        teY = np.concatenate((teY, np.array([[0, 0, 0, 0, 0, 1]])))

    print('Loading test:', teI / len(teList) * 100)
np.delete(teX, 0, 0)
np.delete(teY, 0, 0)
print(trX.shape)
print(trY.shape)
print(teX.shape)
print(teY.shape)


with gzip.open('Marcel-data/train-images-ubyte.gz', 'wb') as f:
    f.write(trX.tobytes())
with gzip.open('Marcel-data/train-labels-ubyte.gz', 'wb') as f:
    f.write(trY.tobytes())
with gzip.open('Marcel-data/test-images-ubyte.gz', 'wb') as f:
    f.write(teX.tobytes())
with gzip.open('Marcel-data/test-labels-ubyte.gz', 'wb') as f:
    f.write(teY.tobytes())

'''
import numpy as np
import gzip

with gzip.open('Marcel-data/train-images-ubyte.gz', 'rb') as f:
    trX = np.frombuffer(f.read())
with gzip.open('Marcel-data/train-labels-ubyte.gz', 'rb') as f:
    trY = np.frombuffer(f.read())
with gzip.open('Marcel-data/test-images-ubyte.gz', 'rb') as f:
    teX = np.frombuffer(f.read())
with gzip.open('Marcel-data/test-labels-ubyte.gz', 'rb') as f:
    teY = np.frombuffer(f.read())

trX = trX.reshape((-1,5016))
trY = trY.reshape((-1,6))
teX = teX.reshape((-1,5016))
teY = teY.reshape((-1,6))

print(trX.shape)
print(trY.shape)
print(teX.shape)
print(teY.shape)
'''
