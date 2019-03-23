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
import random

from tqdm import tqdm

#import the module implementing a neural network that we want to fool
import neuralnet

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

#######################################
#FUNCTIONS DIFFERENTIAL EVOLUTION
#######################################

def function(model, target, img, dict, input):
    row1 = int(input[0])
    col1 = int(input[1])
    r1 = int(input[2])
    g1 = int(input[3])
    b1 = int(input[4])

    row2 = int(input[5])
    col2 = int(input[6])
    r2 = int(input[7])
    g2 = int(input[8])
    b2 = int(input[9])

    row3 = int(input[10])
    col3 = int(input[11])
    r3 = int(input[12])
    g3 = int(input[13])
    b3 = int(input[14])

    row4 = int(input[15])
    col4 = int(input[16])
    r4 = int(input[17])
    g4 = int(input[18])
    b4 = int(input[19])

    row5 = int(input[20])
    col5 = int(input[21])
    r5 = int(input[22])
    g5 = int(input[23])
    b5 = int(input[24])
    '''
    row = int(input[0]*31)
    col = int(input[1]*31)
    r = int(input[2]*255)
    g = int(input[3]*255)
    b = int(input[4]*255)
    '''
    #img = copy.deepcopy(img)

    store1 = copy.deepcopy(img[0][row1][col1])
    store2 = copy.deepcopy(img[0][row2][col2])
    store3 = copy.deepcopy(img[0][row3][col3])
    store4 = copy.deepcopy(img[0][row4][col4])
    store5 = copy.deepcopy(img[0][row5][col5])

    img[0][row1][col1] = r1, g1, b1
    img[0][row2][col2] = r2, g2, b2
    img[0][row3][col3] = r3, g3, b3
    img[0][row4][col4] = r4, g4, b4
    img[0][row5][col5] = r5, g5, b5

    preds = model.predict(img)
    if get_max_class(preds, dict) != target:
        return -1

    img[0][row1][col1] = store1
    img[0][row2][col2] = store2
    img[0][row3][col3] = store3
    img[0][row4][col4] = store4
    img[0][row5][col5] = store5

    #img = img.astype('uint8')
    #plt.imshow(img[0])
    #plt.show()

    return preds[0][target]

def get_random_input():
    x1 = float(randint(0, 31))
    y1 = float(randint(0, 31))
    r1 = float(randint(0, 255))
    g1 = float(randint(0, 255))
    b1 = float(randint(0, 255))

    x2 = float(randint(0, 31))
    y2 = float(randint(0, 31))
    r2 = float(randint(0, 255))
    g2 = float(randint(0, 255))
    b2 = float(randint(0, 255))

    x3 = float(randint(0, 31))
    y3 = float(randint(0, 31))
    r3 = float(randint(0, 255))
    g3 = float(randint(0, 255))
    b3 = float(randint(0, 255))

    x4 = float(randint(0, 31))
    y4 = float(randint(0, 31))
    r4 = float(randint(0, 255))
    g4 = float(randint(0, 255))
    b4 = float(randint(0, 255))

    x5 = float(randint(0, 31))
    y5 = float(randint(0, 31))
    r5 = float(randint(0, 255))
    g5 = float(randint(0, 255))
    b5 = float(randint(0, 255))
    '''
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    '''
    return x1, y1, r1, g1, b1, x2, y2, r2, g2, b2, x3, y3, r3, g3, b3, x4, y4, r4, g4, b4, x5, y5, r5, g5, b5

def new_par(pop, f, limit, i_parameter, population):
    a = pop[randint(0, population - 1)][i_parameter]
    b = pop[randint(0, population - 1)][i_parameter]
    c = pop[randint(0, population - 1)][i_parameter]
    return (a + f*(b - c)) % limit
    '''
    value = (a + f*(b - c))
    if value > 1:
        value = 1
    if value < 0:
        value = 0
    return value
    '''

def differentialAlgorithm(model, target, img, iterations, population, f, range_pixel, range_rgb, dict):
    pop = []

    for p in range(0, population):
        pop.append(get_random_input())

    for i in tqdm(range(0, iterations)):
        #print("Iteration: " + str(i))
        for p in tqdm(range(0, population)):
            #print("Guy: " + str(p))
            new = new_par(pop, f, range_pixel, 0, population), new_par(pop, f, range_pixel, 1, population), new_par(pop, f, range_rgb, 2, population), new_par(pop, f, range_rgb, 3, population), new_par(pop, f, range_rgb, 4, population), \
                new_par(pop, f, range_pixel, 5, population), new_par(pop, f, range_pixel, 6, population), new_par(pop, f, range_rgb, 7, population), new_par(pop, f, range_rgb, 8, population), new_par(pop, f, range_rgb, 9, population), \
                new_par(pop, f, range_pixel, 10, population), new_par(pop, f, range_pixel, 11, population), new_par(pop, f, range_rgb, 12, population), new_par(pop, f, range_rgb, 13, population), new_par(pop, f, range_rgb, 14, population), \
                new_par(pop, f, range_pixel, 15, population), new_par(pop, f, range_pixel, 16, population), new_par(pop, f, range_rgb, 17, population), new_par(pop, f, range_rgb, 18, population), new_par(pop, f, range_rgb, 19, population), \
                new_par(pop, f, range_pixel, 20, population), new_par(pop, f, range_pixel, 21, population), new_par(pop, f, range_rgb, 22, population), new_par(pop, f, range_rgb, 23, population), new_par(pop, f, range_rgb, 24, population)

            value = function(model, target, img, dict, new)
            if value == -1:
                best_int = []
                for i in range(0, len(new)):
                    best_int.append(int(new[i]))
                return best_int
            if value < function(model, target, img, dict, pop[p]):
                pop[p] = new

    best = pop[0]
    for p in range(1, population):
        if function(model, target, img, dict, pop[p]) < function(model, target, img, dict, best):
            best = pop[p]

    best_int = []
    '''
    best_int.append(int(best[0]*31))
    best_int.append(int(best[1]*31))
    best_int.append(int(best[2]*255))
    best_int.append(int(best[3]*255))
    best_int.append(int(best[4]*255))
    '''
    for i in range(0, len(best)):
        best_int.append(int(best[i]))
    return best_int

#function that compute a perturbation, trying to fool the network (True if the algorithm find a solution)
def fool_image(model, img, target, number_of_pixel, budget, show_image, dict):

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

    iterations = 20
    population = 400

    range_pixel = 32
    range_rgb = 256

    f = 0.5

    args = differentialAlgorithm(model, target, copy_input, iterations, population, f, range_pixel, range_rgb, dict)
    print(args)

    #getting results
    #args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)

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
    n_class = str(dict[get_max_class(preds, dict)])
    print("New class: " + n_class)

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

    if show_image:
        plt.show()

    if p_class != n_class:
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
start_img_index = 0 #number of the first image used in cifar10
end_img_index = 1 #last number (NOT incluted)
number_of_pixel = 5 #number of pixel that we will try to change (IT CAN BE: 1, 3, 5)
budget = 100
show_image = False #False = don't show the image
###############################

mispredicted_images = 0
#load model
model = neuralnet.cifar10vgg(False)

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for img_index in range(start_img_index, end_img_index): #image that will be modified

    img = x_test[img_index]

    target = y_test[img_index][0]

    #y_train = keras.utils.to_categorical(y_train, 10) #trasform the class into an array (0 .. 1 ... 0)
    #y_test = keras.utils.to_categorical(y_test, 10)

    res = fool_image(model, img, target, number_of_pixel, budget, show_image, dict)
    print(res)
    if res == True:
        mispredicted_images += 1

print("Number of mis-predicted images: " + str(mispredicted_images))
