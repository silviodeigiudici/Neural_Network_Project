from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import *
import uuid
import numpy as np

class allcnn(Sequential):
    def __init__(self):
        self.seed = 22
        np.random.seed(self.seed)
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        K.set_image_dim_ordering('tf')
        self.is_bn = False
        self.is_dropout = True

        self.model = self.build_model()
        self.model.load_weights("./networks/allcnn/all_cnn_weights_0.9088_0.4994.hdf5")

    def build_model(self):
        # build the network architecture
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(96, 3, 3, border_mode='same'))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2)))
        if self.is_dropout:
            model.add(Dropout(0.5))

        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2, 2)))
        if self.is_dropout:
            model.add(Dropout(0.5))

        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(192, 1, 1, border_mode='valid'))
        if self.is_bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 1, 1, border_mode='valid'))

        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        return model

    def predict(self,x):
        # normalize the images
        x = x.astype('float32')
        x /= 255
        return self.model.predict(x)
