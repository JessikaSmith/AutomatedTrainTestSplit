from data import *

from source.GA import Selector, Mutation, Crossover
from vis_tools import *
from model import QRNN_model

import logging
import numpy as np
import operator
import pandas as pd

dataset = pd.read_csv('/data/GAforAutomatedTrainTestSplit/data/dataset/dataset.csv')
verification = pd.read_csv('/data/GAforAutomatedTrainTestSplit/data/dataset/verification.csv')
qrnn_model = QRNN_model(dataset)


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


# TODO: add optimization
# Fitness function is aimed to evaluate recall, precision
# or f-measure of the performance on the verification dataset

# not to forget to add '-' sign (change max to min)
def _evaluate_fitness(permutation, objective='f1'):
    assert objective in ['recall', 'precision', 'f1']
    # TODO: remove coefficients hardcore
    train_set = dataset.iloc[0:round(len(permutation) * 0.7), :]
    test_set = dataset.iloc[round(len(permutation) * 0.7):, :]
    if objective == 'recall':
        pass
    if objective == 'precision':
        pass
    if objective == 'f1':
        X_train = train_set['text'].tolist()
        y_train = train_set['label'].tolist()
        X_test = test_set['text'].tolist()
        y_test = test_set['text'].tolist()
        qrnn_model.fit(X_train=X_train, y_train=y_train,
                       X_test=X_test, y_test=y_test)
        result = qrnn_model.evaluate_on_verification(verification=verification)
    return result


class Permutation:
    def __init__(self, permutation):
        self.permutation = permutation
        self.fitness = _evaluate_fitness(permutation)
        self._prob = None

    def update_permutation(self, new_permutation):
        self.permutatoin = new_permutation
        self.fitness = _evaluate_fitness(new_permutation)


class GAPipeline:
    def __init__(self, num_of_trials=100, population_size=25, best_perc=0.3,
                 mutation_probability=0.4):
        self.number_of_trials = num_of_trials
        self.population_size = population_size
        self.best_perc = best_perc
        self.mutation_probability = mutation_probability

        # TODO: no need to load it here
        self.size = dataset.shape[0]

        # loggers initialization
        self.run_logger = init_logger('run')

    def _generate_population(self):
        population = []
        for _ in range(self.population_size):
            path = np.random.permutation([i for i in range(self.size)])
            population.append(Permutation(path))
        return population

    # TODO: log best result so far with params
    def run(self):
        self.run_logger.debug('I am running')
        self.run_logger.info('Initializing population...')
        population = self._generate_population()
        s = Selector(selection_type='roulette')
        c = Crossover(crossover_type='ordered')
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
            population = new_generation[:self.population_size]
            for i in range(1, len(population)):
                population[i].update_path(m.mutation(population[i].path,
                                                     mutation_probability=self.mutation_probability))
            population.sort(key=operator.attrgetter('fitness'), reverse=False)
            self.run_logger.info('Generation %s best so far %s' % (ii, population[0].fitness))
            self.run_logger.debug('Best permutation: %s' % (' '.join(population[0].permutation)))
            x.append(ii)
            y.append(population[0].fitness)
        draw_convergence(x, y, 'ps = %s, bp = %s, mr = %s' % (
            round(self.population_size, 2), round(self.best_perc, 2),
            round(self.mutation_probability, 2)))
        return population[0].fitness


ga = GAPipeline()
ga.run()