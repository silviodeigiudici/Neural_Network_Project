from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.datasets import cifar10

import numpy as np

from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib

from PIL import Image

import copy

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



model = VGG16(weights='imagenet', include_top=True)

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



#results = model.predict(x_test)
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''



img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(224, 224))

img.save('my.png')

'''
pixelmap = img.load()
pixelmap[0,0] = (0, 0, 0)
pixelmap[0,1] = (0, 0, 0)
pixelmap[0,2] = (0, 0, 0)
pixelmap[0,3] = (0, 0, 0)
pixelmap[0,4] = (0, 0, 0)
img.save('save.png')


x = image.img_to_array(img)


img = Image.fromarray(x, 'RGB')
#img.save('my.png')
img.show()


x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
'''

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print(decode_predictions(preds, top=3)[0])


i1 = inst.variables.OrderedDiscrete(range(0, 224))
j1 = inst.variables.OrderedDiscrete(range(0, 224))
r1 = inst.variables.OrderedDiscrete(range(0, 255))
g1 = inst.variables.OrderedDiscrete(range(0, 255))
b1 = inst.variables.OrderedDiscrete(range(0, 255))

i2 = inst.variables.OrderedDiscrete(range(0, 224))
j2 = inst.variables.OrderedDiscrete(range(0, 224))
r2 = inst.variables.OrderedDiscrete(range(0, 255))
g2 = inst.variables.OrderedDiscrete(range(0, 255))
b2 = inst.variables.OrderedDiscrete(range(0, 255))

i3 = inst.variables.OrderedDiscrete(range(0, 224))
j3 = inst.variables.OrderedDiscrete(range(0, 224))
r3 = inst.variables.OrderedDiscrete(range(0, 255))
g3 = inst.variables.OrderedDiscrete(range(0, 255))
b3 = inst.variables.OrderedDiscrete(range(0, 255))

i4 = inst.variables.OrderedDiscrete(range(0, 224))
j4 = inst.variables.OrderedDiscrete(range(0, 224))
r4 = inst.variables.OrderedDiscrete(range(0, 255))
g4 = inst.variables.OrderedDiscrete(range(0, 255))
b4 = inst.variables.OrderedDiscrete(range(0, 255))

i5 = inst.variables.OrderedDiscrete(range(0, 224))
j5 = inst.variables.OrderedDiscrete(range(0, 224))
r5 = inst.variables.OrderedDiscrete(range(0, 255))
g5 = inst.variables.OrderedDiscrete(range(0, 255))
b5 = inst.variables.OrderedDiscrete(range(0, 255))


def function_net(i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, model, old_img):
    print("Function")
    img = copy.deepcopy(old_img)

    pixelmap = img.load()
    pixelmap[i1,j1] = (r1, g1, b1)
    pixelmap[i2,j2] = (r2, g2, b2)
    pixelmap[i3,j3] = (r3, g3, b3)
    pixelmap[i4,j4] = (r4, g4, b4)
    pixelmap[i5,j5] = (r5, g5, b5)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0][0][2] #TODO: try to use a vector in order to minimize the best 3 classes of imagenet, not only the first

#ifunc = inst.InstrumentedFunction(myfunction, arg1, arg2)
ifunc = inst.InstrumentedFunction(function_net, i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, model, img)

# create the instrumented function using the "Instrumentation" instance above
#ifunc = inst.instrument(myfunction)
print(ifunc.dimension)  # 5 dimensional space as above


optimizer = optimizerlib.TwoPointsDE(dimension=ifunc.dimension, budget=400)
recommendation = optimizer.optimize(ifunc)

print(recommendation)

args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)

best_i1 = args[0]
best_j1 = args[1]
best_r1 = args[2]
best_g1 = args[3]
best_b1 = args[4]

best_i2 = args[5]
best_j2 = args[6]
best_r2 = args[7]
best_g2 = args[8]
best_b2 = args[9]

best_i3 = args[10]
best_j3 = args[11]
best_r3 = args[12]
best_g3 = args[13]
best_b3 = args[14]

best_i4 = args[15]
best_j4 = args[16]
best_r4 = args[17]
best_g4 = args[18]
best_b4 = args[19]

best_i5 = args[20]
best_j5 = args[21]
best_r5 = args[22]
best_g5 = args[23]
best_b5 = args[24]


pixelmap = img.load()
pixelmap[best_i1,best_j1] = (best_r1, best_g1, best_b1)
pixelmap[best_i2,best_j2] = (best_r2, best_g2, best_b2)
pixelmap[best_i3,best_j3] = (best_r3, best_g3, best_b3)
pixelmap[best_i4,best_j4] = (best_r4, best_g4, best_b4)
pixelmap[best_i5,best_j5] = (best_r5, best_g5, best_b5)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print("Hacked:")
print(decode_predictions(preds, top=3)[0])
img.save('save.png')

print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))
