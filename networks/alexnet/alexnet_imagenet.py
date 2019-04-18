import sys
#sys.path.append('../../convnets-keras/')
sys.path.append('convnets-keras/') #added in order to work with non-targeted
from keras import backend as K
from keras.optimizers import SGD
from convnetskeras.convnets import convnet
from convnetskeras.imagenet_tool import id_to_synset, synset_to_id, id_to_words

class alexnet:
    def __init__(self):
        K.set_image_dim_ordering('th')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #self.model = convnet('alexnet', weights_path="alexnet_weights_imagenet.h5", heatmap=False)
        self.model = convnet('alexnet', weights_path="./networks/alexnet/alexnet_weights_imagenet.h5", heatmap=False)
        self.model.compile(optimizer=sgd, loss='mse')

    def checkInBestFive(self, pred, idx_target):
        best_ids = pred[0].argsort()[::-1][:5]
        return (idx_target in best_ids, list(best_ids))

    def getIdxMaxPred(self, pred):
        best_ids = pred[0].argsort()[::-1][:1]
        return int(best_ids[0])

    def getClassByNum(self, num):
        return str(id_to_synset(num))

    def getNumByClass(self, img_cls):
      return int(synset_to_id(img_cls))

    def getStrByClass(self, img_cls):
      return id_to_words(img_cls)

    def predict(self, elem):
      return self.model.predict(elem)
