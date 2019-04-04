import sys
sys.path.append('../../convnets-keras/')
from keras import backend as K
from keras.optimizers import SGD
from convnetskeras.convnets import convnet
from convnetskeras.imagenet_tool import id_to_synset
import numpy as np

class alexnet:
    def __init__(self):
        K.set_image_dim_ordering('th')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = convnet('alexnet', weights_path="alexnet_weights_imagenet.h5", heatmap=False)
        self.model.compile(optimizer=sgd, loss='mse')
    
    def getIdxMaxPred(self, pred):
        return int(np.where(pred == np.amax(pred))[1])
    
    def getClassByNum(self, num):
        return str(id_to_synset(num))
    
    def predict(self, elem):
      return self.model.predict(elem)
