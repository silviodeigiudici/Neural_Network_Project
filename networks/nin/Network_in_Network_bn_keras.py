import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard

class nin:
    def __init__(self, build=False):
        self.num_classes = 10
        self.weight_decay  = 0.0001
        self.x_shape = [32,32,3]

        if build:
          self.model = self.build_model()
          self.model.load_weights('./networks/nin/nin_bn.h5')
        else:
          self.model = load_model('./networks/nin/nin_bn.h5')

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay
        
        model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
        
        model.add(Dropout(dropout))
        
        model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
        
        model.add(Dropout(dropout))
        
        model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def color_preprocessing(self, x_test):
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

        return x_test

    def predict(self, x):
      x_test  = self.color_preprocessing(x)
      return self.model.predict(x_test)
