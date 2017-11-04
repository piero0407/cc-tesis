from __future__ import print_function

import gzip as gz
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def next_batch(dataset, batch_size, iterator):
    """Return the next `batch_size` examples from this data set."""
    start = batch_size * iterator
    end = batch_size * (iterator + 1)
    return dataset[0][start:end], dataset[1][start:end]


def convolution_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                          name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = pool_shape
    strides = pool_shape
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    out_layer = tf.nn.relu(out_layer)

    return out_layer


def run_cnn(numprints):
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    f = gz.open('dataset.pkl.gz', 'rb')
    training_data, validation_data = pkl.load(f, encoding='latin1')

    # Python optimisation variables
    learning_rate = 0.00001
    epochs = 100
    batch_size = 64

    # declare the training data placeholders
    # input x - for 100 x 100 pixels = 10000 - this is the flattened image data that is drawn from
    # mnist.train.nextbatch()
    x = tf.placeholder(tf.float32, [None, 128 * 128])
    # dynamically reshape the input
    x_shaped = tf.reshape(x, [-1, 128, 128, 1])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 6])

    layer1 = convolution_layer(x_shaped, 1, 32, [5, 5], [
                               1, 4, 4, 1], name='layer1')
    layer2 = convolution_layer(layer1, 32, 64, [3, 3], [
                               1, 2, 2, 1], name='layer2')
    layer3 = convolution_layer(layer2, 64, 128, [3, 3], [
                               1, 2, 2, 1], name='layer3')

    flattened = tf.reshape(layer3, [-1, 8 * 8 * 128])

    wd1 = tf.Variable(tf.truncated_normal(
        [8 * 8 * 128, 1200], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1200], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)

    wd2 = tf.Variable(tf.truncated_normal([1200, 6], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([6], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        np.random.seed(1981)
        tf.set_random_seed(1981)
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(training_data[1]) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = next_batch(training_data, batch_size, i)
                _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            test_acc = sess.run(accuracy,
                                feed_dict={x: validation_data[0], y: validation_data[1]})
            if epoch % numprints == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(
                    avg_cost), "test accuracy: {:.3f}%".format(test_acc * 100))

        acc = sess.run(accuracy, feed_dict={
            x: validation_data[0], y: validation_data[1]})
        print("\nTraining complete!", "{:.3f}%".format(acc * 100))


if __name__ == "__main__":
    run_cnn(10)
