import tensorflow as tf

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = "0"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session

set_session(session)
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import time
import numpy as np

from vis_tools import *

np.random.seed(1337)  # for reproducibility

from model import QRNN
from model.metrics import *
from model import Processor

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dropout
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

import pandas as pd

# TODO: Add this
# from data import Processor

from collections import OrderedDict


class QRNN_model:
    # TODO: reduce the vocabulary if needed
    def __init__(self, dataset, batch_size=128, max_features=80000,
                 max_len=100, emb_dim=300, emb_type='w2v',
                 spatial_dropout=0.1, window_size=3,
                 dropout=0.3, kernel_regularizer=1e-6,
                 bias_regularizer=1e-6, kernel_constraint=6,
                 bias_constraint=6, loss='binary_crossentropy',
                 optimizer='adam', model_type='Bidirectional',
                 lr=0.00005, clipnorm=None, epochs=2,
                 weights=False, trainable=True, previous_weights=None,
                 activation='sigmoid'):
        self.batch_size = batch_size
        self.max_features = max_features
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.p = Processor(max_features=max_features,
                           emb_type=emb_type,
                           max_len=max_len,
                           emb_dim=emb_dim)
        self.get_emb(dataset)
        # params
        self.spatial_dropout = spatial_dropout
        self.window_size = window_size
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.loss = loss
        self.optimizer = optimizer
        self.model_type = model_type
        self.lr = lr
        self.clipnorm = clipnorm
        self.epochs = epochs
        self.weights = weights
        self.trainable = trainable
        self.previous_weights = previous_weights
        self.activation = activation

        self.model = None

    def init_model(self):
        model = Sequential()
        # TODO: decide with embedding matrix
        if self.weights:
            model.add(Embedding(self.max_features,
                                self.emb_dim,
                                weights=[self.p.embedding_matrix],
                                trainable=self.trainable))
        else:
            model.add(Embedding(self.max_features, self.emb_dim))
        model.add(SpatialDropout1D(self.spatial_dropout))
        model.add(Bidirectional(QRNN(self.emb_dim // 2,
                                     window_size=self.window_size,
                                     dropout=self.dropout,
                                     kernel_regularizer=l2(self.kernel_regularizer),
                                     bias_regularizer=l2(self.bias_regularizer),
                                     kernel_constraint=maxnorm(self.kernel_constraint),
                                     bias_constraint=maxnorm(self.bias_constraint))))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation=self.activation))
        # TODO: Add vis if is needed
        if self.clipnorm:
            optimizer = optimizers.Adam(lr=self.lr,
                                        clipnorm=self.clipnorm)
        else:
            optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        #print(model.summary())
        self.model = model

    # TODO: fix dataset name
    def get_emb(self, dataset, dataset_name='negative_employees'):
        dataset = dataset['text'].values.tolist()
        self.p.fit_processor(x_train=dataset, x_train_name=dataset_name)

    # data
    def fit(self, X_train, y_train, X_test, y_test):

        self.init_model()
        timing = str(int(time.time()))

        reduce_rate = ReduceLROnPlateau(monitor='val_loss')
        callbacks_list = [reduce_rate]

        X_train, y_train = self.p.prepare_input(X_train, y_train)
        print('Train params: ', len(X_train), len(y_train))
        X_test, y_test = self.p.prepare_input(X_test, y_test)
        print('Test params: ', len(X_test), len(y_test))

        if not self.weights:
            self.model.fit(X_train, y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=(X_test, y_test),
                           callbacks=callbacks_list)
            # in case of strong will to save the model
            # path_to_weights = '../produced_data/model_%s.h5' % (timing)
            # path_to_architecture = "../produced_data/architecture_%s.h5"
            # self.model.save_weights(path_to_weights)
            # self.model.save(path_to_architecture)
            # print('Model is saved %s' % path_to_weights)

    def evaluate_on_verification(self, verification):
        text = verification.text.tolist()
        prep_verification = self.p.prepare_input(text)
        ver_res = self.model.predict_classes(prep_verification)
        label = verification['label'].tolist()
        ver_res = [i[0] for i in ver_res]
        return calculate_f1(label, ver_res)
