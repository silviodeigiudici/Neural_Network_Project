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

def get_max_class_new(preds, dict):
    index = 0
    max = 0
    index_max = 0
    for v in preds:
        if v > max:
            max = v
            index_max = index
        index += 1
    return index_max

def print_images(images, file):
    f = open(file, "r")
    s = f.read()
    f.close()
    images_string = s.strip().split("\n")
    for image in images_string:
        list = image.strip().split(",")
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

#######################################
#FUNCTIONS DIFFERENTIAL EVOLUTION
#######################################

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

    return x1, y1, r1, g1, b1, x2, y2, r2, g2, b2, x3, y3, r3, g3, b3, x4, y4, r4, g4, b4, x5, y5, r5, g5, b5

def new_par(population, F, limit, i_parameter, num_population, best):
    a = population[randint(0, num_population - 1)][i_parameter]
    b = population[randint(0, num_population - 1)][i_parameter]
    return (best[i_parameter] + F*(a - b)) % limit

def set_image(matrix, p, input):
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


    store1 = copy.deepcopy(matrix[p][row1][col1])
    store2 = copy.deepcopy(matrix[p][row2][col2])
    store3 = copy.deepcopy(matrix[p][row3][col3])
    store4 = copy.deepcopy(matrix[p][row4][col4])
    store5 = copy.deepcopy(matrix[p][row5][col5])


    matrix[p][row1][col1] = r1, g1, b1
    matrix[p][row2][col2] = r2, g2, b2
    matrix[p][row3][col3] = r3, g3, b3
    matrix[p][row4][col4] = r4, g4, b4
    matrix[p][row5][col5] = r5, g5, b5

    return store1, store2, store3, store4, store5


def unset_images(old_pixel_store, matrix, num_population, new_population):
    for i in range(0, num_population):
        store1, store2, store3, store4, store5 = old_pixel_store[i]

        row1 = int(new_population[i][0])
        col1 = int(new_population[i][1])

        row2 = int(new_population[i][5])
        col2 = int(new_population[i][6])

        row3 = int(new_population[i][10])
        col3 = int(new_population[i][11])

        row4 = int(new_population[i][15])
        col4 = int(new_population[i][16])

        row5 = int(new_population[i][20])
        col5 = int(new_population[i][21])

        matrix[i][row1][col1] = store1
        matrix[i][row2][col2] = store2
        matrix[i][row3][col3] = store3
        matrix[i][row4][col4] = store4
        matrix[i][row5][col5] = store5

def new_matrix(img, num_population):
    matrix = np.ndarray((num_population, 32, 32, 3))
    population_indeces = range(0, num_population)
    for p in population_indeces:
        matrix[p] = copy.deepcopy(img[0])
    return matrix

def new_son(population, F, range_pixel, range_rgb, num_population, best):

    row1 = new_par(population, F, range_pixel, 0, num_population, best)
    col1 = new_par(population, F, range_pixel, 1, num_population, best)
    r1 = new_par(population, F, range_rgb, 2, num_population, best)
    g1 = new_par(population, F, range_rgb, 3, num_population, best)
    b1 = new_par(population, F, range_rgb, 4, num_population, best)

    row2 = new_par(population, F, range_pixel, 5, num_population, best)
    col2 = new_par(population, F, range_pixel, 6, num_population, best)
    r2 = new_par(population, F, range_rgb, 7, num_population, best)
    g2 = new_par(population, F, range_rgb, 8, num_population, best)
    b2 = new_par(population, F, range_rgb, 9, num_population, best)

    row3 = new_par(population, F, range_pixel, 10, num_population, best)
    col3 = new_par(population, F, range_pixel, 11, num_population, best)
    r3 = new_par(population, F, range_rgb, 12, num_population, best)
    g3 = new_par(population, F, range_rgb, 13, num_population, best)
    b3 = new_par(population, F, range_rgb, 14, num_population, best)

    row4 = new_par(population, F, range_pixel, 15, num_population, best)
    col4 = new_par(population, F, range_pixel, 16, num_population, best)
    r4 = new_par(population, F, range_rgb, 17, num_population, best)
    g4 = new_par(population, F, range_rgb, 18, num_population, best)
    b4 = new_par(population, F, range_rgb, 19, num_population, best)

    row5 = new_par(population, F, range_pixel, 20, num_population, best)
    col5 = new_par(population, F, range_pixel, 21, num_population, best)
    r5 = new_par(population, F, range_rgb, 22, num_population, best)
    g5 = new_par(population, F, range_rgb, 23, num_population, best)
    b5 = new_par(population, F, range_rgb, 24, num_population, best)

    return row1, col1, r1, g1, b1, row2, col2, r2, g2, b2, row3, col3, r3, g3, b3, row4, col4, r4, g4, b4, row5, col5, r5, g5, b5

def get_best_individual(old_value, pop, num_population):
    best_value = old_value[0]
    best = pop[0]
    population_indeces = range(1, num_population)
    for i in population_indeces:
        if old_value[i][target] < best_value[target]:
            best_value = old_value[i]
            best = pop[i]
    return best

def trasform_to_int(solution):
    best_int = []
    indeces = range(0, len(solution))
    for i in indeces:
        best_int.append(int(solution[i]))
    return best_int

def create_population(img, num_population):
    matrix = new_matrix(img, num_population)
    population = []
    new_population = []
    population_indeces = range(0, num_population)
    for p in population_indeces:
        rand = get_random_input()
        set_image(matrix, p, rand)
        population.append(rand)
        new_population.append(rand)
    return population, new_population, matrix

def differentialAlgorithm(model, target, img, iterations, num_population, F, range_pixel, range_rgb, dict):

    population, new_population, matrix = create_population(img, num_population)

    old_value = model.predict(matrix)

    best = get_best_individual(old_value, population, num_population)

    matrix = new_matrix(img, num_population)

    iterations_indeces = range(0, iterations)
    for i in tqdm(iterations_indeces):
        old_pixel_store = []
        population_indeces = range(0, num_population)
        for p in population_indeces:
            new = new_son(population, F, range_pixel, range_rgb, num_population, best)

            #set_image(matrix, p, new)
            store = set_image(matrix, p, new)
            old_pixel_store.append(store)

            new_population[p] = new

        value = model.predict(matrix)

        #matrix = new_matrix(img, num_population)
        unset_images(old_pixel_store, matrix, num_population, new_population)

        for p in population_indeces:
            if value[p][target] < old_value[p][target]:
                population[p] = new_population[p]
                old_value[p] = value[p]
                if get_max_class_new(old_value[p], dict) != target:
                    print(old_value[p])
                    return trasform_to_int(population[p])

        best = get_best_individual(old_value, population, num_population)

    return trasform_to_int(best)

#function that compute a perturbation, trying to fool the network (True if the algorithm find a solution)
def fool_image(model, img, img_index, target, number_of_pixel, show_image, dict, save, file, iterations, population, f, range_pixel, range_rgb):

    #shows the original image
    plt.imshow(img)
    if show_image:
        plt.show()

    input = np.ndarray((1, 32, 32, 3))
    input[0] = copy.deepcopy(img)

    #x_train = x_train.astype('float32')
    input = input.astype('float32')

    #set copy
    copy_input = copy.deepcopy(input)

    #predict the class of the original image
    original_preds = model.predict(input)

    args = differentialAlgorithm(model, target, copy_input, iterations, population, f, range_pixel, range_rgb, dict)
    print(args)

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
    string = ""
    for k in range(0, number_of_pixel):
        string += ", " + str(args[index]) + ", " + str(args[index + 1]) + ", " + str(args[index + 2]) + ", " + str(args[index + 3]) + ", " + str(args[index + 4])
        print("Pixel: (" + str(args[index + 1]) + ", " + str(args[index]) + ")", end=", ")
        print("Rgb: (" + str(args[index + 2]) + ", " + str(args[index + 3]) + ", " + str(args[index + 4]) + ")")
        index += 5
    string += "\n"

    if show_image:
        plt.show()

    if p_class != n_class:
        if save:
            line = str(img_index) + string
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
start_img_index = 0 #number of the first image used in cifar10
end_img_index = 5 #last number (NOT incluted)
number_of_pixel = 5 #number of pixel that we will try to change (IT CAN BE: 1, 3, 5)
show_image = False #False = don't show the image
save = True #if you want to save the result
num_images = 1 #set the number of images to be extracted
iterations = 50
population = 150
range_pixel = 32
range_rgb = 256
f = 0.5
###############################

mispredicted_images = 0
#load model
model = networks.vgg16.vgg16_cifar10.cifar10vgg()

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print_images(x_test, "save/results_non-targeted.txt")
'''
list = range(start_img_index, end_img_index) #USELESS if you use the random selection:
'''
'''
#random images
list = []
max = len(x_test)
for i in range(0, num_images):
    list.append(randint(0, max))
'''
'''
if save:
    file = open("save/results_non-targeted.txt", "w")

for img_index in list: #image that will be modified

    img = x_test[img_index]

    target = y_test[img_index][0]

    #y_train = keras.utils.to_categorical(y_train, 10) #trasform the class into an array (0 .. 1 ... 0)
    #y_test = keras.utils.to_categorical(y_test, 10)

    res = fool_image(model, img, img_index, target, number_of_pixel, show_image, dict, save, file, iterations, population, f, range_pixel, range_rgb)
    print(res)
    if res == True:
        mispredicted_images += 1

if save:
    file.close()

#use this function if you want to print all the images in the file Results
#print_images(x_test, "save/results_non-targeted.txt")

print("Number of mis-predicted images: " + str(mispredicted_images))
'''
