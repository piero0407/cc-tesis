
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import gzip as gz
import pickle as pkl
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


def weightVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


# In[3]:


def biasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), dtype=tf.float32)


# In[4]:


def forwardPropagation(X, W, b, G):
    L = len(W)
    A = X
    for l in range(1, L):
        Z = tf.matmul(A, W[l]) + b[l]
        A = G[l](Z)
        
    return A


# In[5]:


def initializeParameters(layers):
    L = len(layers)
    W = [None]*L
    b = [None]*L
    for l in range(1, L):
        W[l] = weightVariable([layers[l-1], layers[l]])
        b[l] = weightVariable([1, layers[l]])
        
    return W, b


# In[11]:


def dnnModel(tra, val, #tes,
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
        W, b = initializeParameters(layers)
        X = tf.placeholder(tf.float32, shape=[None, tra['X'].shape[1]])
        Y = tf.placeholder(tf.float32, shape=[None, tra['Y'].shape[1]])
        Y_ = forwardPropagation(X, W, b, activations)
        
        # Reg L2
        regularizer = tf.nn.l2_loss(W[L-1])
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_)
        loss = tf.reduce_mean(loss + lambd * regularizer)
        # loss = tf.reduce_mean(loss)
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
                fd = { X: nextBatch(tra['X']), Y: nextBatch(tra['Y']) }
                sess.run(step, feed_dict=fd)
            if printLoss and (epoch + 1) % interval == 0:
                print('Loss at %4d: %12.6f'%(epoch, sess.run(loss, feed_dict=fd)))
                
        prctAcc = sess.run(accuracy, feed_dict={X: tra['X'], Y: tra['Y']})
        print("Training accuracy: %6.2f%%"%(prctAcc*100))
        prctAcc = sess.run(accuracy, feed_dict={X: val['X'], Y: val['Y']})
        print("Validation accuracy: %6.2f%%"%(prctAcc*100))
        #prctAcc = sess.run(accuracy, feed_dict={X: tes['X'], Y: tes['Y']})
        #print("Test accuracy: %6.2f%%"%(prctAcc*100))


# In[9]:


def loadData():
    f = gz.open('dataset.pkl.gz', 'rb')
    tra, val = pkl.load(f, encoding='latin1')
    f.close()
    tra = {'X': tra[0], 'Y': tra[1]}
    val = {'X': val[0], 'Y': val[1]}
    
    return tra, val


# In[ ]:


def test():
    tra, val = loadData()
    activations = [None,
                   lambda z: tf.nn.relu(z),
                   #lambda z: tf.nn.relu(z),
                   #lambda z: tf.nn.relu(z),
                   lambda z: tf.nn.sigmoid(z)]
    layers = [128*128, 512, 6]
    dnnModel(tra, val, #tes,
              layers,
              activations,
              numIter=600,
              batchSize=64,#tra["X"].shape[0],
              alpha=0.000001,
              numPrints=10)
        
test()

