from keras.datasets import cifar10
###########class associated with each number########
#airplane : 0
#automobile : 1
#bird : 2
#cat : 3
#deer : 4
#dog : 5
#frog : 6
#horse : 7
#ship : 8
#truck : 9
#################

import numpy as np

import matplotlib.pyplot as plt

#nevergrad
from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib
import nevergrad.optimization as optimization

from concurrent import futures

import copy

#import the module implementing a neural network that we want to fool
import neuralnet

#setting data
img_index = 2 #image that will be modified
number_of_pixel = 1 #number of pixel that we will try to change (IT CAN BE: 1, 3, 5)
budget = 1000 #number of iterations
#############

#load model
model = neuralnet.cifar10vgg(False)

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#shows the original image
plt.imshow(x_test[img_index])
plt.show()

#x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

target = y_test[img_index][0]

#y_train = keras.utils.to_categorical(y_train, 10) #trasform the class into an array (0 .. 1 ... 0)
#y_test = keras.utils.to_categorical(y_test, 10)

input = np.ndarray((1, 32, 32, 3))
input[0] = x_test[img_index]

#predict the class of the original image
original_preds = model.predict(input)

#function that return the arguments used by nevergrad
def new_point():
    row = inst.variables.OrderedDiscrete(range(0, 32))
    col = inst.variables.OrderedDiscrete(range(0, 32))
    r = inst.variables.OrderedDiscrete(range(0, 255))
    g = inst.variables.OrderedDiscrete(range(0, 255))
    b = inst.variables.OrderedDiscrete(range(0, 255))
    return row, col, r, g, b

#setting arguments
i1, j1, r1, g1, b1 = new_point()
i2, j2, r2, g2, b2 = new_point()
i3, j3, r3, g3, b3 = new_point()
i4, j4, r4, g4, b4 = new_point()
i5, j5, r5, g5, b5 = new_point()

#function that will be optimized for 1 pixel
def function_net1(i1, j1, r1, g1, b1, target, model, input):
    print("Iteration...")
    img = copy.deepcopy(input)

    img[0][i1][j1] = r1, g1, b1

    preds = model.predict(img)

    return preds[0][target]

#function that will be optimized for 3 pixels
def function_net3(i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, target, model, input):
    print("Iteration...")
    img = copy.deepcopy(input)

    img[0][i1][j1] = r1, g1, b1
    img[0][i2][j2] = r2, g2, b2
    img[0][i3][j3] = r3, g3, b3

    preds = model.predict(img)

    return preds[0][target]

#function that will be optimized for 5 pixels
def function_net5(i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, target, model, input):
    print("Iteration...")
    img = copy.deepcopy(input)

    img[0][i1][j1] = r1, g1, b1
    img[0][i2][j2] = r2, g2, b2
    img[0][i3][j3] = r3, g3, b3
    img[0][i4][j4] = r4, g4, b4
    img[0][i5][j5] = r5, g5, b5

    preds = model.predict(img)

    return preds[0][target]

#setting arguments for the function
if number_of_pixel == 1:
    ifunc = inst.InstrumentedFunction(function_net1, i1, j1, r1, g1, b1, target, model, input)
if number_of_pixel == 3:
    ifunc = inst.InstrumentedFunction(function_net3, i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, target, model, input)
if number_of_pixel == 5:
    ifunc = inst.InstrumentedFunction(function_net5, i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, target, model, input)

#asynch implementation
optim = optimization.registry["TwoPointsDE"](dimension=ifunc.dimension, budget=budget)

with futures.ThreadPoolExecutor(max_workers=optim.num_workers) as executor:
    recommendation = optim.optimize(ifunc, executor=executor)

#synch implementation (not used)
#optimizer = optimizerlib.TwoPointsDE(dimension=ifunc.dimension, budget=2000)
#recommendation = optimizer.optimize(ifunc)

#getting results
args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)

#modify the original image
index = 0
for i in range(0, number_of_pixel):
    row = args[index]
    col = args[index + 1]
    rgb = args[index + 2], args[index + 3], args[index + 4]
    input[0][row][col] = rgb
    index += 5

#prediction of the modified image
preds = model.predict(input)

#print value returned by the network
print()
print("Initial prediction:")
print(original_preds)
print()
print("New prediction:")
print(preds)

#shows the modified image
img = input.astype('uint8')
plt.imshow(img[0])

#print values modified
print()
print("Modified pixels")
index = 0
for k in range(0, number_of_pixel):
    print("Pixel: (" + str(args[index + 1]) + ", " + str(args[index]) + ")", end=", ")
    print("Rgb: (" + str(args[index + 2]) + ", " + str(args[index + 3]) + ", " + str(args[index + 4]) + ")")
    index += 5

plt.show()
