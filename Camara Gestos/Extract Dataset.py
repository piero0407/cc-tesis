import gzip as gz
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np


numOfLabels = 6
randomNum = 25

f = gz.open('marcel.pkl.gz', 'rb')
training_data, validation_data = pkl.load(f, encoding='latin1')

x_train, y_train = training_data
x_validation, y_validation = validation_data

plt.imshow(x_train[randomNum].reshape(128, 128), cmap='gray')
plt.show()
print(y_train[randomNum])

print(training_data[0] .shape)
