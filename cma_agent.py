# Crated by Santiago July 19, 2024
# Updated August 9, 2024
import cma
import numpy as np
import torch
import pickle

from data_loader import load_config_file


class CMAAgent:
    """
    A class to represent a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) agent
    for optimizing decision-making models in electric vehicle routing and charging scenarios.

    Attributes:
        population_size (int): Size of the population used in the evolution strategy.
        max_generation (int): Maximum number of generations for the evolution process.
        es (cma.CMAEvolutionStrategy): Instance of the CMA-ES algorithm.
        weights (list): List to store weights during optimization.
        in_size (int): Dimension of the state input.
        out_size (int): Dimension of the action output.
        num_cars (int): Number of cars involved in the simulation.
        cma_config (dict): Configuration dictionary for the CMA-ES.
        initial_sigma (float): Initial step size for the CMA-ES algorithm.
        model_type (str): Type of model being optimized (e.g., 'optimizer', 'NN_basic').
        states (list): List to store states during optimization.
        actions (list): List to store actions taken during optimization.
        weights_result (list): List to store the resulting weights after optimization.
    """

    def __init__(self, state_dimension, action_dimension, num_cars, seed, agent_index, global_weights):
        """
        Initializes the CMAAgent with configuration and parameters for the CMA-ES algorithm.

        Parameters:
            state_dimension (int): The dimensionality of the input state.
            action_dimension (int): The dimensionality of the output action.
            num_cars (int): The number of cars involved in the scenario.
            seed (int): Seed for random number generation to ensure reproducibility.
            agent_index (int): Index of the agent in a multi-agent scenario.
            global_weights (dict): Pre-trained global weights for initializing the agent's model.
        """
        
        # Load CMA-ES configuration parameters from a YAML file
        fname = 'configs/cma_config.yaml'
        c = load_config_file(fname)
        cma_c = c['cma_parameters']

        # Set up CMA-ES parameters
        self.population_size = cma_c['population_dimension']
        self.max_generation = cma_c['max_generations']
        initial_sigma = cma_c['initial_sigma'] 
        model_type = cma_c['model_type']

        # Initialize random number generator with a seed
        rng = np.random.default_rng(seed)

        # Select and initialize the model type to be optimized with CMA-ES
        if model_type == 'optimizer':
            # Optimizer model does not use state information, directly optimizes weights
            initial_weights = rng.random(action_dimension)
            bounds = [0, 1]
            self.model = self.cma_model

        elif model_type == 'NN_basic':
            # Simple neural network model (initialization to be implemented)
            initial_weights = np.array(rng.random(state_dimension) * 2 - 1)

        else:
            raise RuntimeError('CMA model type not selected or incorrect model in CMA config file.')

        # If global weights are provided, use them to initialize the agent's weights
        if global_weights is not None:
            initial_weights = torch.tensor([tensor for key, tensor in global_weights.items()]).tolist()

        # Initialize the CMA-ES algorithm with the configuration and initial weights
        cma_config = {
            'popsize': self.population_size,
            'maxiter': self.max_generation,
            'bounds': bounds,
            'seed': seed
        }
        es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma, cma_config)

        # Store relevant parameters and objects for the agent
        self.es = es
        self.weights = []
        self.in_size = state_dimension
        self.out_size = action_dimension
        self.num_cars = num_cars
        self.cma_config = cma_config
        self.initial_sigma = initial_sigma
        self.model_type = model_type
        self.states = []
        self.actions = []
        self.weights_result = []

    def cma_model(self, state, weights):
        """
        A simple CMA-ES optimizer model that directly optimizes weights without using the state input.

        Parameters:
            state (ndarray): The current state (not used in this model).
            weights (ndarray): The weights to be optimized.

        Returns:
            ndarray: The optimized weights.
        """
        solutions = weights
        return solutions

    def get_solutions(self):
        """
        Requests a set of candidate solutions (weights) from the CMA-ES algorithm.

        Returns:
            ndarray: A set of candidate solutions for the current generation.
        """
        return self.es.ask()

    def get_best_solutions(self):
        """
        Retrieves the best solution (weights) found by the CMA-ES algorithm so far.

        Returns:
            ndarray: The best weights found during optimization.
        """
        weights = self.es.best.x
        self.weights_result = weights
        return weights

    def get_weights(self):
        """
        Converts the best solution's weights into a dictionary format for easier retrieval.

        Returns:
            dict: A dictionary where keys are weight names and values are the corresponding weights.
        """
        keys = [f'Weight {w}' for w in range(len(self.weights_result))]
        weights = dict(zip(keys, torch.tensor(self.weights_result)))
        return weights

    def tell(self, reward):
        """
        Informs the CMA-ES algorithm about the fitness (reward) of the current generation's solutions.

        Parameters:
            reward (ndarray): The fitness values associated with the solutions of the current generation.
        """
        self.es.tell(self.solutions, reward)

    def save_model(self, fname):
        """
        Saves the current state of the CMA-ES optimizer to a file.

        Parameters:
            fname (str): The file name where the optimizer's state will be saved.
        """
        with open(fname, 'wb') as f:
            pickle.dump(self.es, f)