__author__ = "Aditya Singh"
__version__ = "0.1"

import keras
import numpy as np
np.random.seed(0)
from keras.datasets import cifar100


class CifarGenerator(keras.utils.Sequence):

    def __data_generation(self, index):
        return self.generate_labels(self.data[index*self.batch_size:(index+1)*self.batch_size,], index*self.batch_size)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y

    def __len__(self):
        return int(self.num_samples/self.batch_size)

    def __init__(self, mode='train', batch_size=128):
        self.SIZE = 16
        self.batch_size = batch_size
        if mode == 'train':
            (data, _), ( _, _) = cifar100.load_data()
        else:
            (_, _), (data, _) = cifar100.load_data()
        self.mode = mode
        self.labels = {}
        self.data = data.astype('float32')
        self.num_samples = data.shape[0]

    def generate_labels(self, data, index):
        sample_shape = data[0].shape
        n_rows = int(sample_shape[1]/self.SIZE)
        n_cols = int(sample_shape[0]/self.SIZE)

        ps = np.zeros(shape=(self.batch_size, self.SIZE, self.SIZE, 3))
        ts = np.zeros(shape=(self.batch_size, self.SIZE, self.SIZE, 3))
        ys = np.zeros(shape=(self.batch_size, self.SIZE, self.SIZE, 1))

        for sample in range(self.batch_size):
            r_id = np.random.randint(0, self.SIZE*(n_rows-1))
            c_id = np.random.randint(0, self.SIZE*(n_cols-1))
            t = data[sample][r_id:r_id+self.SIZE, c_id:c_id+self.SIZE,:]

            if np.random.random() < 0.5:
                r_ind = np.random.randint(0, self.batch_size)
            else:
                r_ind = sample

            p = data[r_ind][r_id:r_id+self.SIZE, c_id:c_id+self.SIZE, :]
            if r_ind == sample:
                self.labels[sample+index] = [p, t, 1]
                y = np.ones(shape=(self.SIZE, self.SIZE, 1))
            else:
                y = np.zeros(shape=(self.SIZE, self.SIZE, 1))
                self.labels[sample+index] = [p, t, 0]

            ps[sample, ] = p
            ts[sample, ] = t
            ys[sample, ] = y

        return [ps, ts], ys
