import gzip as gz
import math
import pickle as pkl
import random as ran

import matplotlib.pyplot as plt
import numpy as np

# ran.seed(82364)

f = gz.open('dataset.pkl.gz', 'rb')
training_data, validation_data = pkl.load(f, encoding='latin1')

x_train, y_train = training_data
x_validation, y_validation = validation_data

randomNum = math.trunc(ran.random() * len(x_train))

plt.imshow(x_train[randomNum].reshape(128, 128), cmap='gray')
plt.xlabel(y_train[randomNum])
plt.show()

# print(training_data[0].shape)
# print(training_data[1].shape)
# print(validation_data[0].shape)
# print(validation_data[1].shape)
