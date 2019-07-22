__author__ = "Aditya Singh"
__version__ = "0.1"

import os
import keras
from utils.Agg import cifar100vgg
from utils.small_fcns import FCN
class Model:
    def __init__(self, config):
        self.config = config
        config['checkpoint_path'] = os.path.join(config['checkpoint_path'], config['feature_extractor']
                                                 , config['matching_network'], 'opt{}_lr{}'
                                                 .format(config['optimizer'], config['learning_rate']))
        self.model = None
        self.load()

    def load(self):
        if not os.path.exists(self.config['checkpoint_path']):
            os.makedirs(self.config['checkpoint_path'])
            if self.config['feature_extractor'] == 'mobilenetv2':
                self.fe_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet')
            elif self.config['feature_extractor'] == 'agg':
                self.fe_model = cifar100vgg(train=False).model
            print(self.fe_model.summary())
            if self.fe_model is None:
                raise AssertionError('Cannot load existing fe model from {}'.format(self.config['pretrained_fe_path']))
            if self.config['matching_network'] == 'fcn':
                self.tm_model = FCN().load()
        else:
            self.fe_model = keras.models.load_model(self.config['checkpoint_path'] + '/fe_model.h5')
            print(self.fe_model.summary())
            self.tm_model = keras.models.load_model(self.config['checkpoint_path'] + '/tm_model.h5')
            
            if self.model is None:
                raise AssertionError('Cannot load existing model from {}'.format(self.config['checkpoint_path']))

    def compile_model(self):
        pass
