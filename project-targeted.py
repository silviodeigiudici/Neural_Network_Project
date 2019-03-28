from keras.datasets import cifar10

import numpy as np

import matplotlib.pyplot as plt

#nevergrad
from nevergrad import instrumentation as inst
from nevergrad.optimization import optimizerlib
import nevergrad.optimization as optimization

from concurrent import futures

import copy

from random import randint

#import the module implementing a neural network that we want to fool
import networks.vgg16.vgg16_cifar10

##############################
#SUPPORT FUNCTIONS
#############################

#function that return the class with higher value in preds
def get_max_class(preds, dict):
    index = 0
    max = 0
    index_max = 0
    for v in preds[0]:
        if v > max:
            max = v
            index_max = index
        index += 1
    return index_max

def print_images(images, file):
    f = open(file, "r")
    s = f.read()
    if not s:
      print("\nNo image to show\n")
    else:
      list = s.strip().split(",")
      img_index = int(list[0].strip())
      number_of_pixel = int((len(list) - 1)/5)
      index = 1
      image = images[img_index]
      plt.imshow(image)
      plt.show()
      for i in range(0, number_of_pixel):
          row = int(list[index])
          col = int(list[index + 1])
          rgb = int(list[index + 2]), int(list[index + 3]), int(list[index + 4])
          image[row][col] = rgb
          index += 5
      plt.imshow(image)
      plt.show()
    f.close()

#######################################################################################
#FUNCTIONS NEVERGRAD
########################################################################################

#function that return the arguments used by nevergrad
def new_point():
    row = inst.variables.OrderedDiscrete(range(0, 32))
    col = inst.variables.OrderedDiscrete(range(0, 32))
    r = inst.variables.OrderedDiscrete(range(0, 255))
    g = inst.variables.OrderedDiscrete(range(0, 255))
    b = inst.variables.OrderedDiscrete(range(0, 255))
    return row, col, r, g, b

#function that will be optimized for 1 pixel
def function_net1(i1, j1, r1, g1, b1, target, model, img):
    #print("Iteration...")

    store1 = copy.deepcopy(img[0][i1][j1])

    img[0][i1][j1] = r1, g1, b1

    preds = model.predict(img)

    img[0][i1][j1] = store1

    return -preds[0][target]

#function that will be optimized for 3 pixels
def function_net3(i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, target, model, img):
    #print("Iteration...")

    store1 = copy.deepcopy(img[0][i1][j1])
    store2 = copy.deepcopy(img[0][i2][j2])
    store3 = copy.deepcopy(img[0][i3][j3])

    img[0][i1][j1] = r1, g1, b1
    img[0][i2][j2] = r2, g2, b2
    img[0][i3][j3] = r3, g3, b3

    preds = model.predict(img)

    img[0][i1][j1] = store1
    img[0][i2][j2] = store2
    img[0][i3][j3] = store3

    return -preds[0][target]

#function that will be optimized for 5 pixels
def function_net5(i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, target, model, img):
    #print("Iteration...")

    store1 = copy.deepcopy(img[0][i1][j1])
    store2 = copy.deepcopy(img[0][i2][j2])
    store3 = copy.deepcopy(img[0][i3][j3])
    store4 = copy.deepcopy(img[0][i4][j4])
    store5 = copy.deepcopy(img[0][i5][j5])

    img[0][i1][j1] = r1, g1, b1
    img[0][i2][j2] = r2, g2, b2
    img[0][i3][j3] = r3, g3, b3
    img[0][i4][j4] = r4, g4, b4
    img[0][i5][j5] = r5, g5, b5

    preds = model.predict(img)

    img[0][i1][j1] = store1
    img[0][i2][j2] = store2
    img[0][i3][j3] = store3
    img[0][i4][j4] = store4
    img[0][i5][j5] = store5

    return -preds[0][target]

#function that compute a perturbation, trying to fool the network (True if the algorithm find a solution)
def fool_image(model, img, img_index, target, target_class, number_of_pixel, budget, show_image, dict, save, file):

    #shows the original image
    plt.imshow(img)
    if show_image:
        plt.show()

    input = np.ndarray((1, 32, 32, 3))
    input[0] = img

    #x_train = x_train.astype('float32')
    input = input.astype('float32')

    #set copy
    copy_input = copy.deepcopy(input)

    #predict the class of the original image
    original_preds = model.predict(input)

    #setting arguments
    i1, j1, r1, g1, b1 = new_point()
    i2, j2, r2, g2, b2 = new_point()
    i3, j3, r3, g3, b3 = new_point()
    i4, j4, r4, g4, b4 = new_point()
    i5, j5, r5, g5, b5 = new_point()

    #setting arguments for the function
    if number_of_pixel == 1:
        ifunc = inst.InstrumentedFunction(function_net1, i1, j1, r1, g1, b1, target_class, model, copy_input)
    if number_of_pixel == 3:
        ifunc = inst.InstrumentedFunction(function_net3, i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, target_class, model, copy_input)
    if number_of_pixel == 5:
        ifunc = inst.InstrumentedFunction(function_net5, i1, j1, r1, g1, b1, i2, j2, r2, g2, b2, i3, j3, r3, g3, b3, i4, j4, r4, g4, b4, i5, j5, r5, g5, b5, target_class, model, copy_input)

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
    print("Real class: " + str(dict[target]))

    print()
    p_class = str(dict[get_max_class(original_preds, dict)])
    print("Predicted class: " + p_class)

    print()
    print("New prediction:")
    print(preds)

    print()
    n_class = str(dict[target_class])
    print("Target class: " + n_class)

    print()
    n_class = str(dict[get_max_class(preds, dict)])
    print("New class: " + n_class)

    #shows the modified image
    img = input.astype('uint8')
    plt.imshow(img[0])

    #print values modified
    print()
    print("Modified pixels")
    index = 0
    string = ""
    for k in range(0, number_of_pixel):
        string += ", " + str(args[index]) + ", " + str(args[index + 1]) + ", " + str(args[index + 2]) + ", " + str(args[index + 3]) + ", " + str(args[index + 4])
        print("Pixel: (" + str(args[index + 1]) + ", " + str(args[index]) + ")", end=", ")
        print("Rgb: (" + str(args[index + 2]) + ", " + str(args[index + 3]) + ", " + str(args[index + 4]) + ")")
        index += 5

    if show_image:
        plt.show()

    if p_class != n_class:
        if save:
            line = str(img_index) + string + ", " + str(target_class)
            file.write(line)
        return True
    else:
        return False

##############################################################################
#SETTING UP
#############################################################################

################################
#GLOBAL DATA
#class associated to each number
dict = { 0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
#start_img_index = 2 #number of the first image used in cifar10
#end_img_index = 3 #last number (NOT incluted)
number_of_pixel = 5 #number of pixel that we will try to change (IT CAN BE: 1, 3, 5)
budget = 1500 #number of iterations
show_image = False #False = don't show the image
save = True #if you want to save the result
num_images = 5 #set the number of images to be extracted
target_class = 0
###############################

mispredicted_images = 0
#load model
model = networks.vgg16.vgg16_cifar10.cifar10vgg()

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#list = range(start_img_index, end_img_index) #USELESS if you use the random selection:

#random images
list = []
max = len(x_test)
for i in range(0, num_images):
    list.append(randint(0, max))


for img_index in list: #image that will be modified

    img = x_test[img_index]

    target = y_test[img_index][0]

    #y_train = keras.utils.to_categorical(y_train, 10) #trasform the class into an array (0 .. 1 ... 0)
    #y_test = keras.utils.to_categorical(y_test, 10)

    if save:
        file = open("save/results_targeted.txt", "w")

    res = fool_image(model, img, img_index, target, target_class, number_of_pixel, budget, show_image, dict, save, file)
    print(res)
    if res == True:
        mispredicted_images += 1

    if save:
        file.close()

#use this function if you want to print all the images in the file Results
print_images(x_test, "save/results_targeted.txt")

print("Number of mis-predicted images: " + str(mispredicted_images))
