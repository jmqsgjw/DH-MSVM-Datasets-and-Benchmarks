import math
import random
import time
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cvxopt
import cvxopt.solvers
import mat4py
import matplotlib.pyplot as plt
import numpy
import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential
from keras import regularizers

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


class SELU(object):
    def __init__(self, Trainstep=None, Samplinglength=None):
        self.Trainstep = Trainstep
        self.Samplinglength = Samplinglength

    def Train(self, train_data, train_labels, test_data, test_labels):
        x = train_data
        y = train_labels.T[
            0]
        x_test = test_data
        y_test = test_labels.T[0]
        y = (y + 1) / 2
        y_test = (y_test + 1) / 2
        batchsz = self.Samplinglength
        Dimension = np.size(train_data, axis=1)
        db = tf.data.Dataset.from_tensor_slices((x, y))
        db = db.map(preprocess).shuffle(10000).batch(batchsz)

        db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        db_test = db_test.map(preprocess).batch(batchsz)

        db_iter = iter(db)
        sample = next(db_iter)
        print('batch:', sample[0].shape, sample[1].shape)
        model = Sequential([
            layers.Dense(2 * Dimension, activation=tf.nn.selu, activity_regularizer=regularizers.l2(0.01)),
            # [b, Dimension] => [b, 2*Dimension]
            layers.Dense(2 * Dimension, activation=tf.nn.selu, activity_regularizer=regularizers.l2(0.01)),
            # [b, 2*Dimension] => [b, 2*Dimension]
            layers.Dense(Dimension, activation=tf.nn.selu, activity_regularizer=regularizers.l2(0.01)),
            # [b, Dimension] => [b, Dimension]
            layers.Dense(Dimension, activation=tf.nn.selu, activity_regularizer=regularizers.l2(0.01)),
            # [b, Dimension] => [b, Dimension]
            layers.Dense(2)  # [b, Dimension] => [b, 2]
        ])
        model.build(input_shape=[None, Dimension * 1])
        model.summary()
        # w = w - lr*grad
        optimizer = optimizers.Adam(learning_rate=1e-3)




