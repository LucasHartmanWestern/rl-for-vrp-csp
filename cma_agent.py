import cma
import numpy as np
import torch


class CMAAgent():
    def __init__(self, state_dimension, action_dimension, rng, seed, model_type="Regression"):
        # CMA parameters:
        self.population_size= 100
        self.max_generation = 10
        initial_sigma  = 0.5 

        #Select model to optimize with CMA-ES
        if model_type == 'Regression':
            initial_weights = np.random.randn(action_dimension * state_dimension)
            self.model = self.regresion_model
        elif model_type == 'NN_basic':
            # Generating random initia weights in [-1, 1]
            initial_weights = np.array(rng.random(state_dimension)*2-1)
    
        # Initial weights are enought for cma adjustment
        cma_config = {'popsize': self.population_size, 'maxiter': self.max_generation,\
                      'seed':seed}
        es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma, cma_config)

        # storing information
        self.es = es
        self.weights = []
        self.in_size = state_dimension
        self.out_size= action_dimension
    
    def regresion_model(self, state, weights):
        # Generating a simpler regresion model with random weights for demonstration
        weights_matrix = np.reshape(abs(weights), (self.out_size, self.in_size))
        solutions = np.dot(weights_matrix, state)
        solutions = torch.nn.functional.relu(torch.from_numpy(solutions)).numpy()
        return solutions
        
    def get_actions(self, state, env, fixed_attributes):
        while not self.es.stop():
            population_weights = self.es.ask()
            fitnesses = []
            for weights in population_weights:
                solutions = self.model(state, weights)
                env.generate_paths(solutions, fixed_attributes)
                env.simulate_routes()
                _,_,_,_, rewards = env.get_results()
                fitnesses.append(rewards)
                
            self.es.tell(population_weights, fitnesses)
            self.es.logger.add()
            self.es.disp()

        return self.es.best.x
        
        
        # weights_population = self.es.ask()
        
        # for weights in weights_population:
        #     self.model(state, weights)
        #     env.generate_paths(self.solutions, fixed_attributes)
        #     env.simulate_routes()
        #     _,_,_, rewards = env.get_results()
        # return self.solutions

    def tell(self, reward):
       
        self.es.tell(self.solutions, reward)
        
    