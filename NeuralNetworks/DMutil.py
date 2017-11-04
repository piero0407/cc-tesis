import tensorflow as tf
import numpy as np


def quicktest(creaTensor):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        tensor = creaTensor()
        sess.run(tf.global_variables_initializer())
        print(tensor)
        print(sess.run(tensor))


def weightVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def biasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), dtype=tf.float32)


## uso: quicktest(lambda: algunTensor)