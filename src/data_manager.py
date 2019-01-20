import logging
import pickle
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src
from src.__init__ import *

DATASET_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')
logger = logging.getLogger('emojifier.data_manager')


class EmojifierLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self._n = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d['data'] for d in data])

        self._n = len(images)

        self.images = images.reshape(-1, 48, 48, 1).astype(float) / 255

        self.labels = one_hot(np.hstack([d['labels'] for d in data]), len(EMOTION_MAP))

        return self

    def next_batch(self, batch_size):
        x = self.images[self._i:self._i+batch_size]
        y = self.labels[self._i:self._i+batch_size]

        self._i = (self._i + batch_size) % self._n

        return x, y


class EmojifierDataManager(object):
    def __init__(self):
        logger.info('Loading the dataset ...')
        self.train = EmojifierLoader(['train_batch_'+str(i) for i in range(1)]).load()
        logger.info('Loaded the train-set into the memory !')
        self.test = EmojifierLoader(['test_batch_0']).load()
        logger.info('Loaded the test-set into the memory !')


def unpickle(f):
    with open(os.path.join(DATASET_SAVE_PATH , f), 'rb') as file:
        dictionary = pickle.load(file)

    return dictionary


def one_hot(vec, classes=5):
    n = len(vec)
    one_hot_vec = np.zeros((n, classes))

    one_hot_vec[range(n), vec] = 1

    return one_hot_vec
