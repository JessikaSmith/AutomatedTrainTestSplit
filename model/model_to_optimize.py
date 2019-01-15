import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
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
from model.metrics import calculate_all_metrics
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
    def __init__(self, batch_size=128, max_features=224465,
                 max_len=100, emb_dim=300, emb_type='w2v',
                 spatial_dropout=0.1):
        self.batch_size = batch_size
        self.max_features = max_features
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.p = Processor(max_features=max_features,
                           emb_type=emb_type,
                           max_len=max_len,
                           emb_dim=emb_dim)
        # params
        self.spatial_dropout = spatial_dropout
        self.


    # data
    def fit(self, train, test):
        self.p.fit_processor()


    def evaluate_on_verification(self, verification):
        # TODO: prepare verification sa input
        verification =
        raise NotImplementedError