import numpy as np
import gzip as gz
import pickle as pkl
import matplotlib.pyplot as plt

numOfLabels = 6
randomNum = 100

labes = ['A', 'B', 'C', 'F', 'P', 'V']

f = gz.open('marcel-dataset.pkl.gz', 'rb')
training_data, validation_data = pkl.load(f, encoding='latin1')

x_train = training_data[0]
# y_train = (np.arange(numOfLabels) == training_data[1][:, None]).astype(np.float32)

x_validation = validation_data[0]
# y_validation = (np.arange(numOfLabels) == validation_data[1][:, None]).astype(np.float32)

plt.imshow(x_train[randomNum].reshape(100, 100), cmap='gray')
plt.show()
# print(y_train[randomNum])