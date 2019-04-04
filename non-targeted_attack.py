from keras.datasets import cifar10

import numpy as np

import matplotlib.pyplot as plt

import copy
import time

from random import randint
import random

from tqdm import tqdm

##############################
#SUPPORT FUNCTIONS
#############################

#function that return the class with higher value in preds
def get_max_class(preds):
    index = 0
    max = 0
    index_max = 0
    for v in preds[0]:
        if v > max:
            max = v
            index_max = index
        index += 1
    return index_max

def get_max_class_new(preds):
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
        list_par = image.strip().split(",")
        img_index = int(list_par[0].strip())
        number_of_pixel = int((len(list_par) - 2)/5)
        index = 1
        image = images[img_index]
        plt.imshow(image)
        plt.show()
        for i in range(0, number_of_pixel):
            row = int(list_par[index])
            col = int(list_par[index + 1])
            rgb = int(list_par[index + 2]), int(list_par[index + 3]), int(list_par[index + 4])
            image[row][col] = rgb
            index += 5
        plt.imshow(image)
        plt.show()

#######################################
#FUNCTIONS DIFFERENTIAL EVOLUTION
#######################################

def get_random_perturbation():
    x = float(randint(0, 31))
    y = float(randint(0, 31))
    r = float(randint(0, 255))
    g = float(randint(0, 255))
    b = float(randint(0, 255))
    return x, y, r, g, b

def get_random_input(number_of_pixel):
    perturbations = ()
    indeces = range(0, number_of_pixel)
    for i_pixel in indeces:
        perturbations += get_random_perturbation()
    return perturbations

def new_par(population, F, limit, i_parameter, num_population, best, individual_index, crossover, j, example):

    if random.uniform(0, 1) > crossover and i_parameter != j:
        return example[i_parameter]

    random_list_0 = list(range(0, individual_index))
    random_list_1 = list(range(individual_index + 1, num_population))
    random_list = random_list_0 + random_list_1
    i_a, i_b = random.sample(random_list, 2)
    a = population[i_a][i_parameter]
    b = population[i_b][i_parameter]
    return (best[i_parameter] + F*(a - b)) % limit

def set_image(matrix, p, input, number_of_pixel):
    indeces = range(0, number_of_pixel)
    final_store = ()
    index = 0
    for i_pixel in indeces:
        row = int(input[index])
        col = int(input[index + 1])
        r = int(input[index + 2])
        g = int(input[index + 3])
        b = int(input[index + 4])
        store = copy.deepcopy(matrix[p][row][col])
        matrix[p][row][col] = r, g, b
        final_store += (store,)
        index += 5
    return final_store

def unset_images(old_pixel_store, matrix, num_population, new_population, number_of_pixel):
    population_indeces = range(0, num_population)
    for i in population_indeces:
        indeces = range(0, number_of_pixel)
        index = 0
        for i_pixel in indeces:
            row = int(new_population[i][index])
            col = int(new_population[i][index + 1])
            matrix[i][row][col] = old_pixel_store[i][i_pixel]
            index += 5

def new_matrix(img, num_population):
    matrix = np.ndarray((num_population, 32, 32, 3))
    population_indeces = range(0, num_population)
    for p in population_indeces:
        matrix[p] = copy.deepcopy(img[0])
    return matrix

def new_son(population, F, range_pixel, range_rgb, num_population, best, number_of_pixel, individual_index, crossover, j, example):
    indeces = range(0, number_of_pixel)
    index = 0
    perturbations = ()
    for i_pixel in indeces:
        row = new_par(population, F, range_pixel, index, num_population, best, individual_index, crossover, j, example)
        col = new_par(population, F, range_pixel, index + 1, num_population, best, individual_index, crossover, j, example)
        r = new_par(population, F, range_rgb, index + 2, num_population, best, individual_index, crossover, j, example)
        g = new_par(population, F, range_rgb, index + 3, num_population, best, individual_index, crossover, j, example)
        b = new_par(population, F, range_rgb, index + 4, num_population, best, individual_index, crossover, j, example)
        perturbations += row, col, r, g, b
        index += 5
    return perturbations

def get_best_individual(old_value, pop, num_population, target):
    best_value = old_value[0]
    best = pop[0]
    best_index = 0
    population_indeces = range(1, num_population)
    for i in population_indeces:
        if old_value[i][target] < best_value[target]:
            best_value = old_value[i]
            best = pop[i]
            best_index = i
    return best, best_index

def trasform_to_int(solution):
    best_int = []
    indeces = range(0, len(solution))
    for i in indeces:
        best_int.append(int(solution[i]))
    return best_int

def create_population(img, num_population, number_of_pixel):
    matrix = new_matrix(img, num_population)
    population = []
    new_population = []
    population_indeces = range(0, num_population)
    for p in population_indeces:
        rand = get_random_input(number_of_pixel)
        set_image(matrix, p, rand, number_of_pixel)
        population.append(rand)
        new_population.append(rand)
    return population, new_population, matrix

def differentialAlgorithm(model, target, img, iterations, num_population, F, range_pixel, range_rgb, dict, number_of_pixel, crossover, decrese_crossover):

    population, new_population, matrix = create_population(img, num_population, number_of_pixel)

    old_value = model.predict(matrix)

    best, best_index = get_best_individual(old_value, population, num_population, target)

    matrix = new_matrix(img, num_population)

    iterations_indeces = range(0, iterations)
    for iterations_done in tqdm(iterations_indeces):
        #old_pixel_store = []
        population_indeces = range(0, num_population)
        for p in population_indeces:
            j = randint(0, 5*number_of_pixel)

            new = new_son(population, F, range_pixel, range_rgb, num_population, best, number_of_pixel, p, crossover, j, population[p])

            set_image(matrix, p, new, number_of_pixel)
            #store = set_image(matrix, p, new, number_of_pixel)
            #old_pixel_store.append(store)

            new_population[p] = new

        value = model.predict(matrix)

        matrix = new_matrix(img, num_population)
        #unset_images(old_pixel_store, matrix, num_population, new_population, number_of_pixel)

        old_best_value = old_value[best_index][target]

        for p in population_indeces:
            if value[p][target] < old_value[p][target]:
                population[p] = new_population[p]
                old_value[p] = value[p]
                if get_max_class_new(old_value[p]) != target:
                    print(old_value[p])
                    return trasform_to_int(population[p]), iterations_done

        best, best_index = get_best_individual(old_value, population, num_population, target)

        if old_best_value != old_value[best_index][target]:
            print("New Best!")
            print(old_value[best_index][target])

        crossover -= decrese_crossover

    return trasform_to_int(best), iterations_done

#function that compute a perturbation, trying to fool the network (True if the algorithm find a solution)
def fool_image(model, img, img_index, target, number_of_pixel, show_image, dict, save, file, iterations, population, F, range_pixel, range_rgb, crossover, decrese_crossover):

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
    if get_max_class(original_preds) != target:
        print("\nNetwork mispredict!")
        return True

    args, iterations_done = differentialAlgorithm(model, target, copy_input, iterations, population, F, range_pixel, range_rgb, dict, number_of_pixel, crossover, decrese_crossover)
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
    print("\nInitial prediction:")
    print(original_preds)

    print("\nReal class: " + str(dict[target]))

    p_class = str(dict[get_max_class(original_preds)])
    print("\nPredicted class: " + p_class)

    print("\nNew prediction:")
    print(preds)

    n_class = str(dict[get_max_class(preds)])
    print("\nNew class: " + n_class)

    #shows the modified image
    img = input.astype('uint8')
    plt.imshow(img[0])

    #print values modified
    print("\nModified pixels")
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
            line = str(img_index) + string + ",Success " + n_class + " " + str(iterations_done) + "\n"
            file.write(line)
        return True
    else:
        if save:
            line = str(img_index) + string + ",Fail " + n_class + "\n"
            file.write(line)
        return False

##############################################################################
#SETTING UP
#############################################################################

################################
#GLOBAL DATA
#class associated to each number
dict = { 0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
start_img_index = 1 #number of the first image used in cifar10
end_img_index = 3 #last number (NOT incluted)
number_of_pixel = 1 #number of pixel that we will try to change
show_image = False #False = don't show the image
save = True #if you want to save the result
num_images = 1 #set the number of images to be extracted
iterations = 50
population = 150
crossover = 0.7
decrese_crossover = 0 # 0.5/iterations
range_pixel = 32
range_rgb = 256
F = 0.5
dict_nn = {0: "vgg16",1: "NiN",2: "allcnn"}
neuralnetwork = 0 #0 for vgg16, 1 for nin, 2 for allcnn
###############################

mispredicted_images = 0

#load model
if neuralnetwork == 0:
    import networks.vgg16.vgg16_cifar10
    model = networks.vgg16.vgg16_cifar10.cifar10vgg()
elif neuralnetwork == 1:
    import networks.nin.Network_in_Network_bn_keras
    model = networks.nin.Network_in_Network_bn_keras.nin()
elif neuralnetwork == 2:
    import networks.allcnn.strided_all_CNN_keras
    model = networks.allcnn.strided_all_CNN_keras.allcnn()

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#use this function if you want to print all the images in the file Results
#print_images(x_test, "save/results_non-targeted.txt")

images_list = range(start_img_index, end_img_index) #USELESS if you use the random selection:

'''
#random images
images_list = []
max = len(x_test)
for i in range(0, num_images):
    images_list.append(randint(0, max))
'''

if save:
    file = open("save/non-targeted_saves/results_%d_" % time.time() +str(number_of_pixel)+"_"+dict_nn[neuralnetwork]+".txt", "w")

for img_index in images_list: #image that will be modified

    img = x_test[img_index]

    target = y_test[img_index][0]

    #y_train = keras.utils.to_categorical(y_train, 10) #trasform the class into an array (0 .. 1 ... 0)
    #y_test = keras.utils.to_categorical(y_test, 10)
    res = fool_image(model, img, img_index, target, number_of_pixel, show_image, dict, save, file, \
                    iterations, population, F, range_pixel, range_rgb, crossover, decrese_crossover)
    print(res)
    if res == True:
        mispredicted_images += 1

if save:
    file.close()

print("Number of mis-predicted images: " + str(mispredicted_images))
