from random import randint

def function(input):
    x, y, r, g, b = input
    return x + y + (r - 50)**2 + g + (b - 20)**2

def get_random_input():
    x = float(randint(0, 31))
    y = float(randint(0, 31))
    r = float(randint(0, 255))
    g = float(randint(0, 255))
    b = float(randint(0, 255))
    return x, y, r, g, b

def new_par(pop, f, limit, i_parameter, population, best):
    a = pop[randint(0, population - 1)][i_parameter]
    b = pop[randint(0, population - 1)][i_parameter]
    c = pop[randint(0, population - 1)][i_parameter]
    #return (best[i_parameter] + f*(b - c)) % limit
    return (a + f*(b - c)) % limit

def differentialAlgorithm(iterations, population, f, range_pixel, range_rgb):
    pop = []

    for p in range(0, population):
        pop.append(get_random_input())

    best_result = function(pop[0])
    best = pop[0]
    for i in range(1, population):
        if function(pop[i]) < best_result:
            best_result = function(pop[i])
            best = pop[i]

    for i in range(0, iterations):
        for p in range(0, population):
            example = pop[p]
            new = new_par(pop, f, range_pixel, 0, population, best), new_par(pop, f, range_pixel, 1, population, best), new_par(pop, f, range_rgb, 2, population, best), new_par(pop, f, range_rgb, 3, population, best), new_par(pop, f, range_rgb, 4, population, best)
            if function(new) < function(pop[p]):
                pop[p] = new
        best_result = function(pop[0])
        best = pop[0]
        for i in range(1, population):
            if function(pop[i]) < best_result:
                best_result = function(pop[i])
                best = pop[i]

    best = pop[0]
    for p in range(1, population):
        if function(pop[p]) < function(best):
            best = pop[p]

    best_int = []
    for i in range(0, len(best)):
        best_int.append(int(best[i]))

    return best_int

iterations = 10
population = 100

range_pixel = 32
range_rgb = 256

f = 0.5
print(differentialAlgorithm(iterations, population, f, range_pixel, range_rgb))
