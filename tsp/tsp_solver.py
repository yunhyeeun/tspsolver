import random
import os
import re
import math
import argparse
import csv
import signal
import numpy as np

fitcnt = 0

'''
 City : class about city
 id : city id
 coord, sec : city position
 dist : distance map
'''
class City:
    def __init__ (self, id, coord, sec):
        self.id = id
        self.coord = coord
        self.sec = sec
        self.dist = {}

    def __repr__ (self):
        # return 'City(id = {}, coord = {}, section = {})'.format(self.id, self.coord, self.sec)
        return '{}'.format(self.id)

    def calcDist(self, dst):
        if dst in self.dist:
            return self.dist[dst.id]
        else:
            d = distance(self.coord, dst.coord, self.sec, dst.sec)
            self.dist[dst.id] = d
            return d
    
    def getId(self):
        return self.id

    def getCoord(self):
        return self.coord
    
    def getSec(self):
        return self.sec

class Route:
    def __init__ (self, individual = None, fitness = None):
        self.individual = individual
        self.fitness = fitness
    
    def __repr__ (self):
        # return 'Route: {}\nFitness: {}\n'.format(self.individual, 1 / self.fitness)
        return 'Fitness: {}\n'.format(1 / self.fitness)

    def getIndividual(self):
        return self.individual

    def getFitness(self):
        return self.fitness

    def setIndividual(self, individual):
        self.individual = individual
    
    def setFitness(self, fitness):
        self.fitness = fitness

def signal_handler(signum, frame):
    raise Exception("Timed Out!")

def init():
    parser = argparse.ArgumentParser(description= 'Execute TSP solver...')
    parser.add_argument('inputFile',
                    help='the name of input file')
    parser.add_argument('-p', type=int,
                    help='set the size of the population')
    parser.add_argument('-f', type=int, default=math.inf,
                    help='set the total number of fitness evaluation')
    parser.add_argument('-g', type=int, default=math.inf,
                    help='set the number of generation limit')
    parser.add_argument('-t', type=int, default=300,
                    help='set time limit of execution')
    args = parser.parse_args()
    return args

'''
readFile(FILENAME) : read FILENAME file and return content
'''

def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    result = [ e.strip() for e in lines if e != "" ]
    f.close()
    return result

def writeFile(filename, result):
    f = open(filename, 'w')
    wr = csv.writer(f)
    for i, e in enumerate(result):
        wr.writerow([e.getId() + 1])


'''
findDimension(DATA) : return the number of cities
'''
def findDimension(data):
    p = re.compile(r'\d+')
    for e in data:
        if (e.startswith('DIMENSION')):
            m = p.search(e)
            return int(m.group())

'''
getCities(DATA): return the list of cities(id, position)
'''
def getCities(data):
    cities = []
    for i in range(6, len(data) - 1):
        arr = data[i].split(" ")
        nums = list(map(float, arr))
        cities.append(City(int(nums[0] - 1), nums[1], nums[2]))
    return cities

'''
isCorrectIndividual(INDIVIDUAL) : return TRUE if INDIVIDUAL is correct individual otherwise return FALSE
'''
def isCorrectIndividual(route):
    individual = route.getIndividual()
    arr = np.full(len(individual), False)
    for e in individual:
        if arr[int(e.getId())]:
            print ('duplicate!: {}\n'.format(e.getId()))
            return False
        else:
            arr[e.getId()] = True
    for i in range(len(arr)):
        if arr[i] is False:
            print ('incomplete\n')
            return False
    return True

'''
initPop(CITIES, PSIZE) : make init population with PSIZE
'''
def initPop(cities, pSize):
    population = np.empty(pSize, dtype = object)
    for i in range(pSize):
        individual = random.sample(cities, len(cities))
        population[i] = Route(individual)
        population[i].setFitness(evaluate(population[i]))
    return population

'''
distance(X1, X2, Y1, Y2) : return the distance between (X1, Y1) and (X2, Y2)
'''
def distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (0.5)

'''
evaluate(ROUTE) : return the cost(fitness) of INDIVIDUAL
!IMPORTANT: this function is related to hyperparameter f
'''
def evaluate(route):
    result = 0
    individual = route.getIndividual()
    for i in range(len(individual)):
        if i == len(individual) - 1:
            result += individual[i].calcDist(individual[0])
        else:
            result += individual[i].calcDist(individual[i + 1])
    global fitcnt
    fitcnt += 1
    return 1 / result

'''
getFittest(POPULATION): return fittest individual in POPULATION
'''
def getFittest(population):
    max = 0
    fittest = population[0]
    for i, individual in enumerate(population):
        fitness = individual.getFitness()
        if fitness is None:
            fitness = evaluate(individual)
            individual.setFitness(fitness)
        if (fitness > max):
            max = fitness
            fittest = individual
    return fittest

'''
selection(POPULATION, SUM) : return selected route by proportional selection
'''
def selection(population, sum):
    value = random.uniform(0, sum)
    current = 0
    for individual in population:
        fitness = individual.getFitness()
        current += fitness
        if current > value:
            return individual

'''
tournament(POPULATION, TSIZE) : return fittest individual in POPULATION using tournament selection
'''
def tournament(population, tSize):
    randomPool = random.sample(population, tSize)
    fittest = getFittest(randomPool)
    return fittest

'''
crossover(M, P, CRATE) : create two offsprings from MATERNAL and PATERNAL
'''
    
def crossover(m, p, cRate):
    maternal = m.getIndividual()
    paternal = p.getIndividual()
    dimension = len(maternal)

    num = int(round(cRate * dimension))
    cnt = 0
    points = np.full(dimension, False)

    ## Uniform crossover
    # arr = random.sample(range(dimension), num)
    # for i in arr:
    #     np.put(points, i, True)

    # Two point crossover
    while cnt < num:
        while True:
            start = random.randrange(dimension)
            if not points[start]:
                break
        
        while start < dimension:
            points[start] = True
            cnt += 1
            if cnt == num:
                break
            start += 1

    child1 = np.empty(dimension, dtype=object)
    child2 = np.empty(dimension, dtype=object)
    for i in range(dimension):
        if points[i]:
            np.put(child1, i, maternal[i])
            np.put(child2, i, paternal[i])

    points.fill(False)
    for i in range(dimension):
        if child1[i] is not None:
            np.put(points, int(child1[i].getId()), True)

    # idx = dimension - 1
    # for i in range(dimension):
    #     if not points[int(paternal[i].getId())]:
    #         # while idx < dimension and child1[idx] is not None:
    #         while idx >= 0 and child1[idx] is not None:
    #             idx -= 1
    #         np.put(child1, idx, paternal[i])

    idx = 0
    for i in range(dimension):
        if not points[int(paternal[i].getId())]:
            while idx < dimension and child1[idx] is not None:
                idx += 1
            np.put(child1, idx, paternal[i])

    points.fill(False)
    for i in range(dimension):
        if child2[i] is not None:
            np.put(points, int(child2[i].getId()), True)

    # idx = dimension - 1
    # for i in range(dimension):
    #     if not points[int(maternal[i].getId())]:
    #         # while idx < dimension and child1[idx] is not None:
    #         while idx >= 0 and child2[idx] is not None:
    #             idx -= 1
    #         np.put(child2, idx, maternal[i])
    
    idx = 0
    for i in range(dimension):
        if not points[int(maternal[i].getId())]:
            while idx < dimension and child2[idx] is not None:
                idx += 1
            np.put(child2, idx, maternal[i])

    return [Route(child1), Route(child2)]

'''
mutation(OFFSPRING, MRATE) : mutate OFFSPRING
'''
def mutation(o, mRate):
    offspring = o.getIndividual()
    num = len(offspring)
    points = random.sample(range(num), max(2, int(round(mRate * num)) * 2))
    # points = random.sample(range(num), 2)

    for i in range(0, len(points), 2):
        a = offspring[points[i]]
        b = offspring[points[i + 1]]
        offspring[points[i + 1]] = a
        offspring[points[i]] = b
    o.setIndividual(offspring)
    return o

# test
def run(args):
    # 0. Setup
    # print ('Set up TSP solver...\n')
    data = readFile(args.inputFile)
    dimension = findDimension(data)
    cities = getCities(data)
    # default hyperparameter
    if (args.p is None):
        args.p = max(100, dimension)
        args.p = min(args.p, 1000)
    if (args.f == math.inf):
        if (args.g == math.inf):
            args.f = args.p * 500

    generationcnt = 1

    # 1. Initiate Population
    # print ('Create initial population...\n')
    population = initPop(cities, args.p)
    for i in range(args.p):
        value = evaluate(population[i])
        population[i].setFitness(value)
    population = sorted(population, key=lambda x: x.fitness, reverse=True)

    global fitcnt
    elite = population[0]
    elitecnt = 1
    # print ('Make next generation...\n')

    # Stop when #fitness evaluation exceed OR #generation exceed
    while (fitcnt < args.f and generationcnt <= args.g):
        # print ('Evaluate {}th Generation'.format(generationcnt))
        generationcnt += 1
        # Elitism : maintain the best individual from parents' generation
        eRate = max(1, int(round(args.p * 0.1))) ## TODO : TUNE ELITE RATE IN EMPIRICAL WAY
        nextPop = []
        # print ('Maintain elites to next generation...\n')
        for i in range(eRate):
            nextPop.append(population[i])
        # print ('Start make offsprings...\n')

        # make remain next generation population
        for i in range(0, args.p, 2):
            # 2. Evaluate & choose parent
            tSize = max(5, int(args.p * 0.05)) # TODO : TUNE TOURNAMENT SIZE IN EMPIRICAL WAY
            maternal = population[0]
            paternal = tournament(population, tSize)

            # 3. Crossover
            cRate = 0.5 # TODO : TUNE CROSSOVER RATE IN EMPIRICAL WAY
            offsprings = crossover(maternal, paternal, cRate)

            # 4. Mutation
            mRate = 1 / dimension # TODO : TUNE MUTATION RATE IN EMPIRICAL WAY
            offsprings[0] = mutation(offsprings[0], mRate)
            offsprings[1] = mutation(offsprings[1], mRate)
            
            # 5. Make next generation
            offsprings[0].setFitness(evaluate(offsprings[0]))
            offsprings[1].setFitness(evaluate(offsprings[1]))
            nextPop.append(offsprings[0])
            nextPop.append(offsprings[1])

        nextPop = sorted(nextPop, key=lambda x: x.fitness, reverse=True)

        # To escape local optima, make next generation non-elite
        if (elite.getFitness() == nextPop[0].getFitness()):
            elitecnt += 1
            if elitecnt > 10: # TODO: TUNE ELITECOUNT IN EMPIRICAL WAY
                population = nextPop[len(nextPop) - args.p:]
                elite = population[0]
                elitecnt = 0
            else:
                population = nextPop[:args.p]
                elite = population[0]
        else:
            population = nextPop[:args.p]
            elite = population[0]

        # print ('Elite: {}\n'.format(population[0:10]))
    
    print(1 / population[0].getFitness())
    writeFile('solution.csv', population[0].getIndividual())

if __name__ == '__main__':
    args = init()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(args.t)
    try:
        run(args)

    except Exception as e:
        print (e)
        print ('parameter inputFile : {}'.format(args.inputFile))
        print ('parameter population size : {}'.format(args.p))
        print ('parameter #fitness evaluation : {}'.format(args.f))
        print ('parameter #generation limit : {}'.format(args.g))
        print ('parameter time limit : {}'.format(args.t))


