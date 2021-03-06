# coding: utf-8

# In[1]:


import os

import numpy as np
import tensorflow as tf

from dataset import *
from DMutil import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


def forwardPropagation(X, W, b, G):
    L = len(W)
    A = X
    for l in range(1, L):
        Z = tf.matmul(A, W[l]) + b[l]
        A = G[l](Z)

    return A


# In[3]:


def initializeParameters(layers):
    L = len(layers)
    W = [None]*L
    b = [None]*L
    for l in range(1, L):
        W[l] = weightVariable([layers[l-1], layers[l]])
        b[l] = biasVariable([1, layers[l]])

    return W, b


# In[4]:

def saveParameters(WK, bK, W, b, sess):
    L = len(W)
    LC = len(WK)
    sW = [None]*L
    sb = [None]*L
    sWK = [None]*LC
    sbK = [None]*LC
    for l in range(1, L):
        sW[l] = sess.run(W[l])
        sb[l] = sess.run(b[l])

    for l in range(1, LC):
        sWK[l] = sess.run(WK[l])
        sbK[l] = sess.run(bK[l])

    parameters = (sW, sb, sWK, sbK)
    f = gz.open('parametersINSERTARFECHA.pkl.gz')
    pkl.dump(f, parameters)
    f.close()


def dnnModel(tra, val, #tes,
             convLayers,
             layers,
             activations,
             numIter=10,
             batchSize=128,
             alpha=0.5,
             lambd=0.01,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-08,
             printLoss=True,
             numPrints=10):
    interval = int(numIter / numPrints)
    m = tra['X'].shape[0]
    L = len(layers)

    nextBatch = lambda data: data[offset:offset+batchSize]

    with tf.Session() as sess:
        np.random.seed(1981)
        tf.set_random_seed(1981)
        X = tf.placeholder(tf.float32, shape=[None, tra['X'].shape[1]])
        Y = tf.placeholder(tf.float32, shape=[None, tra['Y'].shape[1]])

        fil = 128
        col = 128
        CL = len(convLayers)

        WK = [None]*CL
        bK = [None]*CL
        for l in range(1, CL):
            WK[l] = weightVariable([convLayers[l][0], convLayers[l][0], convLayers[l-1][1], convLayers[l][1]])
            bK[l] = biasVariable([1, convLayers[l][1]])

        h_pool = tf.reshape(X, [-1, fil, col, 1])
        for l in range(1, CL):
            h_conv = tf.nn.relu(tf.nn.conv2d(h_pool, WK[l], strides=[1, 1, 1, 1], padding='SAME') + bK[l])
            h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            fil = fil // 2
            col = col // 2

        h_flat = tf.reshape(h_pool, [-1, fil*col*convLayers[CL-1][1]])

        W, b = initializeParameters(layers)
        Y_ = forwardPropagation(h_flat, W, b, activations)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_)
        # regularizer = tf.nn.l2_loss(W[L-1])
        # loss = tf.reduce_mean(loss + lambd * regularizer)
        loss = tf.reduce_mean(loss)
        # step = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
        step = tf.train.AdamOptimizer(learning_rate=alpha,
                                      beta1=beta1,
                                      beta2=beta2,
                                      epsilon=epsilon).minimize(loss)

        prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        sess.run(tf.global_variables_initializer())
        for epoch in range(numIter):
            p = np.random.permutation(m)
            tra['X'], tra['Y'] = tra['X'][p], tra['Y'][p]
            for offset in range(0, m, batchSize):
                fd = {X: nextBatch(tra['X']), Y: nextBatch(tra['Y'])}
                loss4print, _ = sess.run([loss, step], feed_dict=fd)
            if printLoss and (epoch + 1) % interval == 0:
                print('Loss at %4d: %12.6f' % (epoch, loss4print))

        saveParameters(WK, bK, W, b, sess)
        prctAcc = sess.run(accuracy, feed_dict={X: tra['X'][:500, :], Y: tra['Y'][:500, :]})
        print("Training accuracy: %6.2f%%" % (prctAcc*100))
        prctAcc = sess.run(accuracy, feed_dict={X: val['X'], Y: val['Y']})
        print("Validation accuracy: %6.2f%%" % (prctAcc*100))
        # prctAcc = sess.run(accuracy, feed_dict={X: tes['X'], Y: tes['Y']})
        # print("Test accuracy: %6.2f%%"%(prctAcc*100))


# In[5]:


def test():
    tra, val = loadDataset()
    activations = [None,
                   lambda z: tf.nn.relu(z),
                   lambda z: tf.nn.relu(z),
                   lambda z: tf.nn.softmax(z)]
    convLayers = [[1, 1], [7, 128], [5, 64], [5, 32]]
    layers = [16*16*32, 512, 256, 6]
    # convLayers = [[1, 1], [5, 32]]
    # layers = [64*64*32, 512, 6]
    dnnModel(tra, val,
             convLayers,
             layers,
             activations,
             numIter=40,
             batchSize=64,
             # batchSize=tra['X'].shape[0],
             alpha=0.00001,
             numPrints=20)

test()
