# Created by Lucas April 11, 2025
import numpy as np
import torch
import pickle
import random
from collections import deque
from data_loader import load_config_file

GRAMMAR = {
    "Network": [["Layer", "Network"], ["Layer"]],
    "Layer": [["dense", "(", "units", ")", "activation"]],
    "units": [["4"], ["8"], ["16"], ["32"], ["64"], ["128"], ["256"]],
    "activation": [["relu"], ["tanh"], ["sigmoid"], ["linear"]]
}



def decode_genotype(genotype_str, input_dim, output_dim):
    """
    Decode a genotype string into a PyTorch neural network model.
    This simplified decoder assumes a genotype composed of tokens such as:
        dense ( units ) activation
    It constructs a sequential model of dense layers.
    """
    layers = []
    tokens = genotype_str.split()
    prev_units = input_dim
    i = 0
    # Look for occurrences of the pattern: dense ( units ) activation
    while i < len(tokens):
        if tokens[i] == "dense":
            if i + 4 < len(tokens) and tokens[i + 1] == "(" and tokens[i + 3] == ")":
                try:
                    units = int(tokens[i + 2])
                except ValueError:
                    units = 32  # default if conversion fails
                act = tokens[i + 4] if i + 4 < len(tokens) else "relu"
                layers.append(torch.nn.Linear(prev_units, units))
                if act == "relu":
                    layers.append(torch.nn.ReLU())
                elif act == "tanh":
                    layers.append(torch.nn.Tanh())
                elif act == "sigmoid":
                    layers.append(torch.nn.Sigmoid())
                prev_units = units
                i += 5
            else:
                i += 1
        else:
            i += 1
            
    layers.append(torch.nn.Linear(prev_units, output_dim))
    # Add final Sigmoid activation to guarantee outputs are in [0, 1]
    layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)


class DenserAgent:
    """
    A DENSER agent for evolving neural network architectures using
    a grammar-based representation (genotype) that is decoded into a PyTorch model.
    
    The agent supports two model types:
      - 'optimizer': which in this simplified example uses continuous vectors
      - 'NN_basic': a neural network with a structured (grammar-based) representation
    """

    def __init__(self, state_dimension, action_dimension, num_cars, seed, agent_index, global_weights, experiment_number):
        """
        Initializes the DenserAgent by loading configuration, setting parameters, and
        initializing the population.
        """
        fname = f'experiments/Exp_{experiment_number}/config.yaml'
        c = load_config_file(fname)
        # We reuse the CMA parameters section for DENSER configuration.
        denser_c = c['cma_parameters']

        self.population_size = denser_c['population_dimension']
        self.max_generation = denser_c['max_generations']
        self.mutation_rate = denser_c.get('mutation_rate', 0.2)
        self.crossover_rate = denser_c.get('crossover_rate', 0.5)
        model_type = denser_c['model_type']

        # Set random seeds for reproducibility.
        # random.seed(seed+agent_index)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed+agent_index)

        self.population = [] # List of individuals (each is a dict with 'genotype', 'structure', and 'fitness')
        self.fitness_history = deque(maxlen=10)

        # Initialize population based on model type.
        if model_type == 'optimizer':
            self.initialize_optimizer_population(action_dimension, seed)
            # In this branch, we assume a dummy DENSER model (e.g., identity).
            self.model = self.denser_model  
        elif model_type == 'NN_basic':
            self.initialize_nn_population(state_dimension, action_dimension, seed)
        else:
            raise RuntimeError('DENSER model type not selected or incorrect in config file.')

        # Initialize best individual.
        if global_weights is not None:
            # In a complete implementation, global_weights could be integrated into the best individual.
            self.best_individual = {
                'genotype': None,
                'structure': None,
                'fitness': float('-inf')
            }
        else:
            self.best_individual = {
                'genotype': None,
                'structure': None,
                'fitness': float('-inf')
            }

        self.in_size = state_dimension
        self.out_size = action_dimension
        self.num_cars = num_cars
        self.denser_config = denser_c
        self.model_type = model_type
        self.states = []
        self.actions = []
        self.weights_result = []
        self.solutions = None

    def generate_random_genotype(self, grammar, symbol="Network"):
        """
        Recursively generate a random genotype (string) from a given grammar.
        """
        if symbol not in grammar:
            return symbol  # terminal symbol
        choices = np.array(grammar[symbol], dtype=object)
        rule = self.rng.choice(choices)
        return " ".join(self.generate_random_genotype(grammar, sym) for sym in rule)
    
    def initialize_optimizer_population(self, action_dimension, seed):
        """
        Initialize population for an "optimizer" model.
        """
        for _ in range(self.population_size):
            # Use a random genotype (even if structure is not used in this branch).
            genotype = self.generate_random_genotype(GRAMMAR, "Network")
            individual = {
                'genotype': genotype,
                'structure': None,  # Not used in optimizer branch.
                'weights': self.rng.random(action_dimension),
                'fitness': float('-inf')
            }
            self.population.append(individual)

    def initialize_nn_population(self, state_dimension, action_dimension, seed):
        """
        Initialize population for a neural network model.
        Each individual is represented by a genotype (string) and its decoded PyTorch model.
        """
        for _ in range(self.population_size):
            genotype = self.generate_random_genotype(GRAMMAR, "Network")
            structure = decode_genotype(genotype, state_dimension, action_dimension)
            individual = {
                'genotype': genotype,
                'structure': structure,
                'fitness': float('-inf')
            }
            self.population.append(individual)

    def denser_model(self, state, individual):
        """
        A placeholder for the optimizer model branch (if using continuous weights).
        """
        return individual['weights']

    def denser_restart(self):
        """
        Restart the evolution process by creating a new population centered on the best individual.
        """
        if self.best_individual['genotype'] is not None:
            self.population = []
            for _ in range(self.population_size):
                # Clone best genotype and apply a small mutation.
                tokens = self.best_individual['genotype'].split()
                print(f'agent denser line 182 tokens {tokens}')
                if tokens:
                    idx = random.randint(0, len(tokens)-1)
                    if tokens[idx] in GRAMMAR:
                        replacement = random.choice(GRAMMAR[tokens[idx]])
                        tokens[idx] = replacement
                mutated_genotype = " ".join(tokens)
                new_structure = decode_genotype(mutated_genotype, self.in_size, self.out_size)
                individual = {
                    'genotype': mutated_genotype,
                    'structure': new_structure,
                    'fitness': float('-inf')
                }
                self.population.append(individual)

    def get_solutions(self):
        """
        Return the list of genotypes (candidate solutions) from the current population.
        """
        self.solutions = [ind['genotype'] for ind in self.population]
        return self.solutions

    def get_best_solutions(self):
        """
        Return the best genotype found so far.
        """
        if self.best_individual['genotype'] is not None:
            self.weights_result.append(self.best_individual['genotype'])
            return self.best_individual['genotype']
        return None

    def get_weights(self):
        """
        For a NN model, return the state dictionary of the best individual's network.
        """
        if self.best_individual['structure'] is not None:
            return self.best_individual['structure'].state_dict()
        return None

    def tell(self, rewards):
        for i, reward in enumerate(rewards):
            self.population[i]['fitness'] = reward
            # For minimization, update if the candidate's fitness is lower than the current best.
            if reward < self.best_individual['fitness']:
                self.best_individual = {
                    'genotype': self.population[i]['genotype'],
                    'structure': self.population[i]['structure'],
                    'fitness': reward
                }
        self.fitness_history.append(min(rewards))  # Optionally use min() for minimization.
        self.evolve_population()


    def evolve_population(self):
        """
        Evolve the population using elitism, tournament selection,
        grammar-based crossover, and mutation.
        """
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        elite_size = max(1, int(0.1 * self.population_size))
        new_population = sorted_population[:elite_size]

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(sorted_population)
            parent2 = self.tournament_selection(sorted_population)
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                # Clone parent1 if no crossover
                child = parent1.copy()
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def tournament_selection(self, population, tournament_size=3):
        """
        Selects an individual from the population using tournament selection.
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness']).copy()

    def crossover(self, parent1, parent2):
        """
        Performs a simple single-point crossover on the genotype strings of two parents.
        Inherits the structure by decoding the new genotype.
        """
        tokens1 = parent1['genotype'].split()
        tokens2 = parent2['genotype'].split()
        if len(tokens1) < 2 or len(tokens2) < 2:
            return parent1.copy()
        cp1 = random.randint(1, len(tokens1) - 1)
        cp2 = random.randint(1, len(tokens2) - 1)
        child_tokens = tokens1[:cp1] + tokens2[cp2:]
        child_genotype = " ".join(child_tokens)
        child_structure = decode_genotype(child_genotype, self.in_size, self.out_size)
        return {
            'genotype': child_genotype,
            'structure': child_structure,
            'fitness': float('inf')
        }

    def mutate(self, individual):
        """
        Mutates an individual by randomly changing one token in the genotype that corresponds to a nonterminal.
        After mutation, the genotype is re-decoded into a network model.
        """
        tokens = individual['genotype'].split()
        indices = [i for i, tok in enumerate(tokens) if tok in GRAMMAR]
        if indices:
            idx = random.choice(indices)
            replacement = random.choice(random.choice(GRAMMAR[tokens[idx]]))
            tokens[idx] = replacement
        new_genotype = " ".join(tokens)
        new_structure = decode_genotype(new_genotype, self.in_size, self.out_size)
        individual['genotype'] = new_genotype
        individual['structure'] = new_structure
        return individual

    def save_model(self, fname):
        """
        Saves the current population, best individual, fitness history, and generation number.
        """
        state = {
            'population': self.population,
            'best_individual': self.best_individual,
            'fitness_history': list(self.fitness_history),
            'generation': len(self.weights_result)
        }
        with open(fname, 'wb') as f:
            pickle.dump(state, f)

    def evaluate_individual(self, individual, train_loader, val_loader, device):
        """
        Evaluates an individual's phenotype (the decoded network) by training it for a few epochs
        and computing the fitness (here, negative total validation loss).
        """
        model = individual['structure'].to(device)
        criterion = torch.nn.MSELoss()  # Choose an appropriate loss based on your task.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 3  # For a proxy evaluation—full training would take longer.
        
        # Train phase (simplified)
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        # Validation phase
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        # Define fitness as negative validation loss so that lower loss gives higher fitness.
        fitness = -total_loss
        individual['fitness'] = fitness
        return fitness

    # (Optional) A method to run the denser model for an input state.
    def run(self, state, individual_index=0):
        """
        Runs the network of a given individual on the provided state.
        """
        if self.model_type == 'NN_basic':
            individual = self.population[individual_index]
            return individual['structure'](torch.tensor(state, dtype=torch.float32))
        else:
            # For 'optimizer' branch, simply return weights (dummy implementation)
            individual = self.population[individual_index]
            return individual['weights']

# Example usage:
if __name__ == "__main__":
    # These variables should be set according to your actual scenario.
    state_dim = 10
    action_dim = 2
    num_cars = 5
    seed = 42
    agent_index = 0
    global_weights = None
    experiment_number = 1

    agent = DenserAgent(state_dim, action_dim, num_cars, seed, agent_index, global_weights, experiment_number)
    
    # Example: Retrieve the solutions (genotypes) of the current population.
    solutions = agent.get_solutions()
    print("Initial Genotypes:")
    for sol in solutions:
        print(sol)
    
    # To evolve, evaluate your individuals with your training and validation loaders,
    # call agent.tell() with the fitness rewards (list of rewards per individual).
    # For demonstration, we can simulate rewards:
    fake_rewards = [random.uniform(-10, 0) for _ in range(agent.population_size)]
    agent.tell(fake_rewards)
    
    # Retrieve best solution genotype and its weights (state_dict if NN model).
    best_solution = agent.get_best_solutions()
    print("\nBest Genotype so far:")
    print(best_solution)
    
    # Save the current agent state
    agent.save_model("denser_agent_state.pkl")
