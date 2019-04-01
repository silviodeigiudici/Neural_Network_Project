import random

def function(input):
    x, y, r, g, b = input
    return x + y + (r - 50)**2 + g + (b - 20)**2

def get_random_input():
    x = float(random.randint(0, 31))
    y = float(random.randint(0, 31))
    r = float(random.randint(0, 255))
    b = float(random.randint(0, 255))
    g = float(random.randint(0, 255))
    # x = random.uniform(0,1)*31
    # y = random.uniform(0,1)*31
    # r = random.uniform(0,1)*255
    # g = random.uniform(0,1)*255
    # b = random.uniform(0,1)*255
    return x, y, r, g, b

def new_par(pop, f, limit, i_parameter, population, best, best_index, p, j, cr):

    if random.uniform(0,1) > cr and i_parameter != j:
        return pop[p][i_parameter]

    #range_list = range(0, population) #puoi escludere best da questa lista (l'indice)!!! cosi non viene beccato
    list_0 = range(0, p)
    list_1 = range(p+1, population)
    range_list = list(list_0) + list(list_1)
    i_a, i_b = random.sample(range_list, 2)
    #a = pop[random.randint(0, population - 1)][i_parameter]
    #b = pop[random.randint(0, population - 1)][i_parameter]
    a = pop[i_a][i_parameter]
    b = pop[i_b][i_parameter]
    return (best[i_parameter] + f*(a - b)) % limit
    #return (a + f*(b - c)) % limit

def differentialAlgorithm(iterations, population, f, range_pixel, range_rgb):
    pop = []

    for p in range(0, population):
        pop.append(get_random_input())

    best_result = function(pop[0])
    best = pop[0]
    best_index = 0
    for i in range(1, population):
        if function(pop[i]) < best_result:
            best_result = function(pop[i])
            best = pop[i]
            best_index = i

    cr = 1

    for i in range(0, iterations):
        for p in range(0, population):
            example = pop[p]
            j = random.randint(0, 5)
            new = new_par(pop, f, range_pixel, 0, population, best, best_index, p, j, cr), \
                new_par(pop, f, range_pixel, 1, population, best, best_index, p, j, cr), \
                new_par(pop, f, range_rgb, 2, population, best, best_index, p, j, cr), \
                new_par(pop, f, range_rgb, 3, population, best, best_index, p, j, cr), \
                new_par(pop, f, range_rgb, 4, population, best, best_index, p, j, cr)
            if function(new) < function(pop[p]):
                pop[p] = new
        best_result = function(pop[0])
        best = pop[0]
        best_index = 0
        for i in range(1, population):
            if function(pop[i]) < best_result:
                best_result = function(pop[i])
                best = pop[i]
                best_index = i
        #cr -= 0.01

    print(cr)
    best_int = []
    for i in range(0, len(best)):
        best_int.append(int(best[i]))

    return best_int

iterations = 15
population = 40

range_pixel = 32
range_rgb = 256

f = 0.5
print(differentialAlgorithm(iterations, population, f, range_pixel, range_rgb))
