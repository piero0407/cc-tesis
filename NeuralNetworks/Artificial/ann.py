import tensorflow as tf
import pickle as pkl
import gzip
import numpy as np
import matplotlib.pyplot as plt

f = gzip.open(r'C:\Users\piero\Desktop\Working\Tesis\mnist.pkl.gz', 'rb')
train_data, validation_data, test_data = pkl.load(f, encoding='latin1')
f.close()

print('Num x (imagenes)', len(train_data[0]))
print('    Num inputs', len(train_data[0][0]))
print('Num y (etiquetas)', len(train_data[1]))
print('    Num outputs', 1)

'''unaimg = train_data[0][5].reshape((28, 28))
plt.imshow(unaimg)
plt.show()
print('Etiqueta correspondiente: ', train_data[1][5])'''

dataset = train_data[0]
labels = train_data[1]
labels = (np.arange(10) == labels[:, None]).astype(np.float32)
vdataset = validation_data[0]
vlabels = validation_data[1]
vlabels = (np.arange(10) == vlabels[:, None]).astype(np.float32)
tdataset = test_data[0]
tlabels = test_data[1]
tlabels = (np.arange(10) == tlabels[:, None]).astype(np.float32)

print('Ejemplo de etiqueta en posicion 0', labels[5])

print('Training data', dataset.shape, labels.shape)
print('Validation data', vdataset.shape, vlabels.shape)
print('Testing data', vdataset.shape, vlabels.shape)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=(None, 28 * 28))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.global_variables_initializer())

bloque = 128
steps = 10001
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(steps):
        idx = step % int(len(dataset) / bloque)
        idx2 = step % int(len(vdataset) / bloque)
        acc, l, y2 = sess.run([accuracy, cross_entropy, y_conv],
                              {x: dataset[idx * bloque:(idx + 1) * bloque, :],
                                  y_: labels[idx * bloque:(idx + 1) * bloque, :],
                                  keep_prob: 0.5})
        if step % 1000 == 0:
            print('accuracy: %.1f%%' % acc)
