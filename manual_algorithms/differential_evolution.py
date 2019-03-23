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

def new_par(pop, f, limit, i_parameter, population):
    a = pop[randint(0, population - 1)][i_parameter]
    b = pop[randint(0, population - 1)][i_parameter]
    c = pop[randint(0, population - 1)][i_parameter]
    return (a + f*(b - c)) % limit

def differentialAlgorithm(iterations, population, f, range_pixel, range_rgb):
    pop = []

    for p in range(0, population):
        pop.append(get_random_input())

    for i in range(0, iterations):
        for p in range(0, population):
            example = pop[p]
            new = new_par(pop, f, range_pixel, 0, population), new_par(pop, f, range_pixel, 1, population), new_par(pop, f, range_rgb, 2, population), new_par(pop, f, range_rgb, 3, population), new_par(pop, f, range_rgb, 4, population)
            if function(new) < function(pop[p]):
                pop[p] = new

    best = pop[0]
    for p in range(1, population):
        if function(pop[p]) < function(best):
            best = pop[p]

    best_int = []
    for i in range(0, len(best)):
        best_int.append(int(best[i]))

    return best_int

iterations = 50
population = 200

range_pixel = 32
range_rgb = 256

f = 0.5
print(differentialAlgorithm(iterations, population, f, range_pixel, range_rgb))
