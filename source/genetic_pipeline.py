from data import *

import logging
import numpy as np


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


class GAPipeline:
    def __init__(self, num_of_trials=100, population_size=25):
        self.number_of_trials = num_of_trials
        self.population_size = population_size

        # TODO: no need to load it here
        self.dataset = load_data('../data/dataset/dataset.csv')
        self.size = self.dataset.shape[0]

        # loggers initialization
        self.run_logger = init_logger('run')

    def _generate_population(self):
        population = []
        for _ in range(self.population_size):
            path = np.random.permutation([i for i in range(self.size)])
            population.append(path)
        return population

    def run(self):
        self.run_logger.debug('I am running')
        self.run_logger.info('Initializing population...')
        population = self._generate_population()


    def objective(self):
        raise NotImplementedError
