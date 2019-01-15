from data import *

from source.GA import Selector, Mutation, Crossover
import logging
import numpy as np
import operator
import random


def init_logger(function_name):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%d-%m-%Y %H:%M',
        filename='../logs/genetic_pipe.log',
        filemode='a'
    )
    logger = logging.getLogger(function_name)
    return logger

def _evaluate_fitness(self):
    raise NotImplementedError

class Permutation:
    def __init__(self, permutation):
        self.permutation = permutation
        self.fitness = _evaluate_fitness(permutation)
        self._prob = None

    def update_permutation(self, new_permutation):
        self.permutatoin = new_permutation
        self.fitness = _evaluate_fitness(new_permutation)

class GAPipeline:
    def __init__(self, num_of_trials=100, population_size=25, best_perc=0.3):
        self.number_of_trials = num_of_trials
        self.population_size = population_size
        self.best_perc = best_perc

        # TODO: no need to load it here
        self.dataset = load_data('../../data/dataset/dataset.csv')
        self.size = self.dataset.shape[0]

        # loggers initialization
        self.run_logger = init_logger('run')

    def _generate_population(self):
        population = []
        for _ in range(self.population_size):
            path = np.random.permutation([i for i in range(self.size)])
            population.append(path)
        return population

    # TODO: log best result so far with params
    def run(self):
        self.run_logger.debug('I am running')
        self.run_logger.info('Initializing population...')
        population = self._generate_population()
        s = Selector(selection_type='roulette')
        c = Crossover(selection_type='ordered')
        m = Mutation(mutation_type='rsm')
        x, y = [], []
        for ii in range(self.number_of_trials):
            population.sort(key=operator.attrgetter('fitness'), reverse=False)
            new_generation = []
            for i in range(int(self.population_size * self.best_perc)):
                new_generation.append(population[i])
            pairs_generator = s.selection(population=population, best_perc=self.best_perc)
        for i, j in pairs_generator:
            child_1, child_2 = c.crossover(parent_1=i.path, parent_2=j.path)
            new_generation.append(Permutation(child_1))
            new_generation.append(Permutation(child_2))


