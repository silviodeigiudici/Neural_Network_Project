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

def new_par(pop, f, limit, i_parameter, population, best):
    a = pop[randint(0, population - 1)][i_parameter]
    b = pop[randint(0, population - 1)][i_parameter]
    return (best[i_parameter] + f*(a - b)) % limit

def set_image(pop_array, p, input):
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


    store1 = copy.deepcopy(pop_array[p][row1][col1])
    store2 = copy.deepcopy(pop_array[p][row2][col2])
    store3 = copy.deepcopy(pop_array[p][row3][col3])
    store4 = copy.deepcopy(pop_array[p][row4][col4])
    store5 = copy.deepcopy(pop_array[p][row5][col5])


    pop_array[p][row1][col1] = r1, g1, b1
    pop_array[p][row2][col2] = r2, g2, b2
    pop_array[p][row3][col3] = r3, g3, b3
    pop_array[p][row4][col4] = r4, g4, b4
    pop_array[p][row5][col5] = r5, g5, b5

    return store1, store2, store3, store4, store5


def unset_images(pop_store, pop_array, population, new_pop):
    for i in range(0, population):
        store1, store2, store3, store4, store5 = pop_store[i]

        row1 = int(new_pop[i][0])
        col1 = int(new_pop[i][1])

        row2 = int(new_pop[i][5])
        col2 = int(new_pop[i][6])

        row3 = int(new_pop[i][10])
        col3 = int(new_pop[i][11])

        row4 = int(new_pop[i][15])
        col4 = int(new_pop[i][16])

        row5 = int(new_pop[i][20])
        col5 = int(new_pop[i][21])

        pop_array[i][row1][col1] = store1
        pop_array[i][row2][col2] = store2
        pop_array[i][row3][col3] = store3
        pop_array[i][row4][col4] = store4
        pop_array[i][row5][col5] = store5

def new_matrix(img, population):
    pop_array = np.ndarray((population, 32, 32, 3))
    population_indeces = range(0, population)
    for p in population_indeces:
        pop_array[p] = copy.deepcopy(img[0])
    return pop_array

def new_son(pop, f, range_pixel, range_rgb, population, best):

    row1 = new_par(pop, f, range_pixel, 0, population, best)
    col1 = new_par(pop, f, range_pixel, 1, population, best)
    r1 = new_par(pop, f, range_rgb, 2, population, best)
    g1 = new_par(pop, f, range_rgb, 3, population, best)
    b1 = new_par(pop, f, range_rgb, 4, population, best)

    row2 = new_par(pop, f, range_pixel, 5, population, best)
    col2 = new_par(pop, f, range_pixel, 6, population, best)
    r2 = new_par(pop, f, range_rgb, 7, population, best)
    g2 = new_par(pop, f, range_rgb, 8, population, best)
    b2 = new_par(pop, f, range_rgb, 9, population, best)

    row3 = new_par(pop, f, range_pixel, 10, population, best)
    col3 = new_par(pop, f, range_pixel, 11, population, best)
    r3 = new_par(pop, f, range_rgb, 12, population, best)
    g3 = new_par(pop, f, range_rgb, 13, population, best)
    b3 = new_par(pop, f, range_rgb, 14, population, best)

    row4 = new_par(pop, f, range_pixel, 15, population, best)
    col4 = new_par(pop, f, range_pixel, 16, population, best)
    r4 = new_par(pop, f, range_rgb, 17, population, best)
    g4 = new_par(pop, f, range_rgb, 18, population, best)
    b4 = new_par(pop, f, range_rgb, 19, population, best)

    row5 = new_par(pop, f, range_pixel, 20, population, best)
    col5 = new_par(pop, f, range_pixel, 21, population, best)
    r5 = new_par(pop, f, range_rgb, 22, population, best)
    g5 = new_par(pop, f, range_rgb, 23, population, best)
    b5 = new_par(pop, f, range_rgb, 24, population, best)

    return row1, col1, r1, g1, b1, row2, col2, r2, g2, b2, row3, col3, r3, g3, b3, row4, col4, r4, g4, b4, row5, col5, r5, g5, b5

def get_best_individual(old_value, pop, population):
    best_value = old_value[0]
    best = pop[0]
    population_indeces = range(1, population)
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

def create_population(img, population):
    pop_array = new_matrix(img, population)
    pop = []
    new_pop = []
    population_indeces = range(0, population)
    for p in population_indeces:
        rand = get_random_input()
        set_image(pop_array, p, rand)
        pop.append(rand)
        new_pop.append(rand)
    return pop, new_pop, pop_array

def differentialAlgorithm(model, target, img, iterations, population, f, range_pixel, range_rgb, dict):

    pop, new_pop, pop_array = create_population(img, population)

    old_value = model.predict(pop_array)

    best = get_best_individual(old_value, pop, population)

    pop_array = new_matrix(img, population)

    iterations_indeces = range(0, iterations)
    for i in tqdm(iterations_indeces):
        pop_store = []
        population_indeces = range(0, population)
        for p in population_indeces:
            new = new_son(pop, f, range_pixel, range_rgb, population, best)

            #set_image(pop_array, p, new)
            store = set_image(pop_array, p, new)
            pop_store.append(store)

            new_pop[p] = new

        value = model.predict(pop_array)

        #pop_array = new_matrix(img, population)
        unset_images(pop_store, pop_array, population, new_pop)

        for p in population_indeces:
            if value[p][target] < old_value[p][target]:
                pop[p] = new_pop[p]
                old_value[p] = value[p]
                if get_max_class_new(old_value[p], dict) != target:
                    print(old_value[p])
                    return trasform_to_int(pop[p])

        best = get_best_individual(old_value, pop, population)

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
        file = open("save/results_non-targeted.txt", "w")

    res = fool_image(model, img, img_index, target, number_of_pixel, show_image, dict, save, file, iterations, population, f, range_pixel, range_rgb)
    print(res)
    if res == True:
        mispredicted_images += 1

    if save:
        file.close()

#use this function if you want to print all the images in the file Results
#print_images(x_test, "save/results_non-targeted.txt")

print("Number of mis-predicted images: " + str(mispredicted_images))
