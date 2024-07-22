import cma
import numpy as np
import torch


class CMAAgent():
    def __init__(self, state_dimension, action_dimension, num_cars, seed, agent_index):
        # CMA parameters:
        self.population_size= 10
        self.max_generation = 2
        initial_sigma  = 0.5
        model_type = 'Regression'

        # Seeding rng
        rng = np.random.default_rng(seed)

        #Select model to optimize with CMA-ES
        if model_type == 'Regression':
            initial_weights = rng.random(action_dimension * state_dimension)
            self.model = self.regresion_model
        elif model_type == 'NN_basic':
            # Generating random initia weights in [-1, 1]
            initial_weights = np.array(rng.random(state_dimension)*2-1)
    
        # Initial weights are enought for cma adjustment
        # cma_config = {'maxiter': self.max_generation, 'seed':seed}
        cma_config = {'popsize': self.population_size*num_cars, 'maxiter': self.max_generation,\
                      'seed':seed}
        es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma, cma_config)

        # storing information
        self.es = es
        self.weights = []
        self.in_size = state_dimension
        self.out_size= action_dimension
        self.num_cars= num_cars
        self.cma_config = cma_config
        self.initial_sigma = initial_sigma
        self.weights = []
        self.model_type = model_type
        self.states = []
        self.actions= []


    def reset_episode(self, reset_weights):
        es = cma.CMAEvolutionStrategy(reset_weights, self.initial_sigma, self.cma_config)

        # storing information
        self.es = es
        self.states = []
        self.actions= []
        
    def regresion_model(self, state, weights):
        # Generating a simpler regresion model with random weights for demonstration
        weights_matrix = np.reshape(weights, (self.out_size, self.in_size))
        solutions = np.dot(weights_matrix, state)
        # solutions = torch.nn.functional.sigmoid(torch.from_numpy(solutions)).numpy()
        #Temporla fix: Limiting simgoid output to not be either 1 or 0
        solutions = 0.9999*(torch.nn.functional.sigmoid(torch.from_numpy(solutions)).numpy()+0.0001)
        return solutions
        
    # def get_actions(self, state, env):
    #     env.cma_store()
    #     print(f'')
    #     while not self.es.stop():
    #         population_weights  = self.es.ask()
    #         population_solutions= self.model(state, population_weights)
    #         fitnesses = []
    #         for cma_idx, solutions in enumerate(population_solutions):
    #             env.cma_copy_store()
    #             env.generate_paths(solutions, None)
    #             env.simulate_routes()
    #             _,_,_,_, rewards = env.get_results()
    #             fitnesses.append(-1*rewards.sum())

    #         self.es.tell(population_weights, fitnesses)
    #         self.es.logger.add()
    #         self.es.disp()

    #     env.cma_clean()
    #     return self.es.best.x


    def run_scenarios(self, env):
        #running cma-es scenarios
        env.cma_store()
        while not self.es.stop():
            
            scenario_solutions = self.es.ask()
            # Reshape the 1D array solution to solutions by car matrix shape
            matrix_solutions = np.reshape(scenario_solutions, (self.population_size, self.num_cars, -1))
            
            fitnesses = np.empty((self.population_size, self.num_cars))
            for pop_idx in range(self.population_size):
                env.cma_copy_store()
                
                for car_idx  in range(self.num_cars):
                    state = env.reset_agent(car_idx)
                    weights = matrix_solutions[pop_idx, car_idx,:]
                    car_route = self.regresion_model(state, weights)
                    env.generate_paths(car_route, None)
                    self.states.append(state)
                    self.actions.append(car_route)

                env.simulate_routes()
                _,_,_,_, rewards = env.get_results()
                fitnesses[pop_idx] = -1*rewards

            self.es.tell(scenario_solutions, fitnesses.flatten())
            self.es.logger.add()
            self.es.disp()
                
        env.cma_clean()
        self.weights_result = self.es.best.x
        return self.weights_result, True

    def run_routes(self, env):

        #running all EV routes in envrionment given the actions from CMA
        pseudo_distributions = []
        for car_idx  in range(self.num_cars):
            state = env.reset_agent(car_idx)
            weights = self.weights_result
            car_route = self.regresion_model(state, weights)
            env.generate_paths(car_route, None)
            pseudo_distributions.append(weights)

        return pseudo_distributions
                
    def get_weights(self):
        keys = [f'Weight {w}' for w in range(len(self.weights_result))]
        weights = dict(zip(keys,torch.tensor(self.weights_result)))
        return weights

    def get_run_info(self):
        return self.actions, self.states
        
    def load_global_weights(self, global_weights):
        # Initial weights are enought for cma adjustment
        weights = torch.tensor([tensor for key, tensor in global_weights.items()]).tolist()
        self.es = cma.CMAEvolutionStrategy(weights, self.initial_sigma, self.cma_config)
        

    def tell(self, reward):
       
        self.es.tell(self.solutions, reward)
        
    