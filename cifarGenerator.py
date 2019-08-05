__author__ = "Aditya Singh"
__version__ = "0.1"

import keras
import numpy as np
from keras.datasets import cifar100
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from utils.utils import bb_intersection_over_union
import cv2
np.random.seed(0)

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


def get_pretrained_model(num_channels=3):
    mv2 = keras.applications.mobilenet.MobileNet(alpha=1.0, input_tensor=Input(shape=(None, None, num_channels)),
                                                 include_top=False, weights='imagenet', pooling=None)
    return mv2


def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


def model_output(model, layer_name, inp):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    org_shape = inp.shape
    intermediate_output = intermediate_layer_model.predict(x=np.reshape(inp, newshape=(-1, org_shape[0], org_shape[1], org_shape[2])))
    return intermediate_output


class CifarFeatureGenerator(keras.utils.Sequence):

    def __data_generation(self, index):
        return self.generate_labels(self.data[index])

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y

    def __len__(self):
        return int(self.num_samples/self.batch_size)

    def __init__(self, mode='train', batch_size=1, generation_stage='output'):
        self.model = get_pretrained_model()
        self.reduction_factor = 32
        self.generation_stage = generation_stage
        self.channels = self.model.output.shape[-1]
        self.batch_size = batch_size

        if mode == 'train':
            (data, _), ( _, _) = cifar100.load_data()
        else:
            (_, _), (data, _) = cifar100.load_data()
        self.mode = mode
        self.labels = {}
        self.data = data.astype('float32')
        self.num_samples = data.shape[0]

    def generate_labels(self, data):
        org_sample_shape = data.shape
        res_data = cv2.resize(data, (3*org_sample_shape[0], 3*org_sample_shape[1]), interpolation=cv2.INTER_CUBIC)
        sample_shape = res_data.shape
        row_size = np.random.choice([32, 64])
        col_size = np.random.choice([32, 64])
        r_row = np.random.choice(sample_shape[0]-row_size+1)
        r_col = np.random.choice(sample_shape[1]-col_size+1)

        template = res_data[r_row:r_row+row_size, r_col:r_col+col_size, :]

        p_f_out = model_output(self.model, self.generation_stage, res_data)
        p_f_shape = p_f_out.shape

        t_f_out = model_output(self.model, self.generation_stage, template)
        t_f_shape = t_f_out.shape

        ys = np.zeros(shape=(self.batch_size, 1))

        shape_diff = [i-j for i, j in zip(p_f_shape, t_f_shape)]

        gt_row = np.random.choice(shape_diff[1])
        gt_col = np.random.choice(shape_diff[2])

        matching_parent = p_f_out[:, gt_row:gt_row+t_f_shape[1], gt_col:gt_col+t_f_shape[2], :]

        approx_gt_positions = (self.reduction_factor*gt_col, self.reduction_factor*gt_row,
                          self.reduction_factor*(gt_col+t_f_shape[2]), self.reduction_factor*(gt_row+t_f_shape[0]))
        iou = bb_intersection_over_union(approx_gt_positions, (r_col, r_row, r_col+col_size, r_row+row_size))
        if iou > 0.25:
            ys[0, ] = 1
        else:
            ys[0, ] = 0
        return [matching_parent, t_f_out], ys
