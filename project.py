from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.datasets import cifar10

import numpy as np

'''
#model = VGG16(weights='imagenet', include_top=False)
#model = VGG16(weights='path_to_weight', include_top=False)
model = VGG16(weights=None, include_top=True, input_shape=(32, 32, 3))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=64, shuffle=True)

print("Done")

results = model.predict(x_test, batch_size=16)
'''


'''
model = VGG16(weights='imagenet', include_top=True)

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''

#results = model.predict(x_test)
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''


'''
img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

'''

from nevergrad import instrumentation as inst

arg1 = inst.variables.OrderedDiscrete([1, 2])
arg2 = inst.variables.OrderedDiscrete([4, 3])

def myfunction(arg1, arg2):
    return (arg1 - arg2)**2

ifunc = inst.InstrumentedFunction(myfunction, arg1, arg2)

# create the instrumented function using the "Instrumentation" instance above
#ifunc = inst.instrument(myfunction)
print(ifunc.dimension)  # 5 dimensional space as above

from nevergrad.optimization import optimizerlib

optimizer = optimizerlib.TwoPointsDE(dimension=ifunc.dimension, budget=100)
recommendation = optimizer.optimize(ifunc)

print(recommendation)

args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)

print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))

def square(x):
    return sum((x - .5)**2)

optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100)
# alternatively, you can use optimizerlib.registry which is a dict containing all optimizer classes
recommendation = optimizer.optimize(square)

print(recommendation)
