import sys
sys.path.append('../../convnets-keras/')
from keras import backend as K
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet

class alexnet:
    def __init__(self):
        K.set_image_dim_ordering('th')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = convnet('alexnet', weights_path="alexnet_weights_imagenet.h5", heatmap=False)
        self.model.compile(optimizer=sgd, loss='mse')
        self.predict(['../../convnets-keras/examples/dog.jpg'])
    
    def predict(self, x):
        x = preprocess_image_batch(x,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
        print(self.model.predict(x))
        
print("Ciaooooo")
model = alexnet()
