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

def new_par_complete(pop, best, f, limit, population):
    a = pop[randint(0, population - 1)]
    b = pop[randint(0, population - 1)]
    c = pop[randint(0, population - 1)]
    d = pop[randint(0, population - 1)]
    prob_mut = 0.7
    new = copy.deepcopy(best)
    lun = len(new)
    range_pixel = 32
    range_rgb = 256
    if random.uniform(0, 1) <= prob_mut:
        new[0] = (new[0] + f*(a[0] + b[0] - c[0] - d[0])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[1] = (new[1] + f*(a[1] + b[1] - c[1] - d[1])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[2] = (new[2] + f*(a[2] + b[2] - c[2] - d[2])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[3] = (new[3] + f*(a[3] + b[3] - c[3] - d[3])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[4] = (new[4] + f*(a[4] + b[4] - c[4] - d[4])) % range_rgb

    if random.uniform(0, 1) <= prob_mut:
        new[5] = (new[5] + f*(a[5] + b[5] - c[5] - d[5])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[6] = (new[6] + f*(a[6] + b[6] - c[6] - d[6])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[7] = (new[7] + f*(a[7] + b[7] - c[7] - d[7])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[8] = (new[8] + f*(a[8] + b[8] - c[8] - d[8])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[9] = (new[9] + f*(a[9] + b[9] - c[9] - d[9])) % range_rgb

    if random.uniform(0, 1) <= prob_mut:
        new[10] = (new[10] + f*(a[10] + b[10] - c[10] - d[10])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[11] = (new[11] + f*(a[11] + b[11] - c[11] - d[11])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[12] = (new[12] + f*(a[12] + b[12] - c[12] - d[12])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[13] = (new[13] + f*(a[13] + b[13] - c[13] - d[13])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[14] = (new[14] + f*(a[14] + b[14] - c[14] - d[14])) % range_rgb

    if random.uniform(0, 1) <= prob_mut:
        new[15] = (new[15] + f*(a[15] + b[15] - c[15] - d[15])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[16] = (new[16] + f*(a[16] + b[16] - c[16] - d[16])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[17] = (new[17] + f*(a[17] + b[17] - c[17] - d[17])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[18] = (new[18] + f*(a[18] + b[18] - c[18] - d[18])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[19] = (new[19] + f*(a[19] + b[19] - c[19] - d[19])) % range_rgb

    if random.uniform(0, 1) <= prob_mut:
        new[20] = (new[20] + f*(a[20] + b[20] - c[20] - d[20])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[21] = (new[21] + f*(a[21] + b[21] - c[21] - d[21])) % range_pixel
    if random.uniform(0, 1) <= prob_mut:
        new[22] = (new[22] + f*(a[22] + b[22] - c[22] - d[22])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[23] = (new[23] + f*(a[23] + b[23] - c[23] - d[23])) % range_rgb
    if random.uniform(0, 1) <= prob_mut:
        new[24] = (new[24] + f*(a[24] + b[24] - c[24] - d[24])) % range_rgb
    '''
    for i in range(0, lun):
        if random.uniform(0, 1) <= prob_mut:
            new[i] = (new[i] + f*(a[i] + b[i] - c[i] - d[i])) % range_pixel
    '''
    return new

def differentialAlgorithm(model, target, img, iterations, population, f, range_pixel, range_rgb, dict):
    pop = []

    for p in range(0, population):
        pop.append(get_random_input())

    best = list(pop[0])
    value_best = function(model, target, img, dict, pop[0])

    for i in tqdm(range(0, iterations)):
        #print("Iteration: " + str(i))
        for p in tqdm(range(0, population)):
            #print("Guy: " + str(p))

            new = new_par_complete(pop, best, f, range_pixel, population)

            '''
            new = new_par(pop, f, range_pixel, 0, population), new_par(pop, f, range_pixel, 1, population), new_par(pop, f, range_rgb, 2, population), new_par(pop, f, range_rgb, 3, population), new_par(pop, f, range_rgb, 4, population), \
                new_par(pop, f, range_pixel, 5, population), new_par(pop, f, range_pixel, 6, population), new_par(pop, f, range_rgb, 7, population), new_par(pop, f, range_rgb, 8, population), new_par(pop, f, range_rgb, 9, population), \
                new_par(pop, f, range_pixel, 10, population), new_par(pop, f, range_pixel, 11, population), new_par(pop, f, range_rgb, 12, population), new_par(pop, f, range_rgb, 13, population), new_par(pop, f, range_rgb, 14, population), \
                new_par(pop, f, range_pixel, 15, population), new_par(pop, f, range_pixel, 16, population), new_par(pop, f, range_rgb, 17, population), new_par(pop, f, range_rgb, 18, population), new_par(pop, f, range_rgb, 19, population), \
                new_par(pop, f, range_pixel, 20, population), new_par(pop, f, range_pixel, 21, population), new_par(pop, f, range_rgb, 22, population), new_par(pop, f, range_rgb, 23, population), new_par(pop, f, range_rgb, 24, population)
            '''

            value = function(model, target, img, dict, new)
            if value == -1:
                best_int = []
                for i in range(0, len(new)):
                    best_int.append(int(new[i]))
                return best_int
            if value < function(model, target, img, dict, pop[p]):
                pop[p] = new
            if value < value_best:
                value = value_best
                best = list(new)

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

    iterations = 10
    population = 50

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
end_img_index = 3 #last number (NOT incluted)
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
