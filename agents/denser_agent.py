# Created by Lucas April 11, 2025
import numpy as np
import torch
import pickle
from collections import deque
from data_loader import load_config_file

GRAMMAR = {
    "Network": [["Layer", "Network"], ["Layer"]],
    "Layer": [["dense", "(", "units", ")", "activation"]],
    # "units": [["4"], ["8"], ["16"], ["32"], ["64"], ["128"], ["256"]],
    "units": [["4"], ["8"], ["16"], ["32"]],
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
                    units = 32
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

    def __init__(self, state_dimension, action_dimension, num_cars, seed, agent_index, global_weights, experiment_number, device):
        """
        Initializes the DenserAgent by loading configuration, setting parameters, and
        initializing the population.
        """
        fname = f'experiments/Exp_{experiment_number}/config.yaml'
        c = load_config_file(fname)
        # Reuse the CMA parameters section for DENSER configuration.
        denser_c = c['cma_parameters']

        self.population_size = denser_c['population_dimension']
        self.max_generation = denser_c['max_generations']
        self.mutation_rate = denser_c.get('mutation_rate', 0.4)
        self.crossover_rate = denser_c.get('crossover_rate', 0.7)
        self.device = device
        model_type = denser_c['model_type']

        # Set random seeds for reproducibility.
        self.rng = np.random.default_rng(seed+agent_index)

        self.population = []
        self.fitness_history = deque(maxlen=10)

        if model_type == 'optimizer':
            self.initialize_optimizer_population(action_dimension, seed)
            self.model = self.denser_model  
        elif model_type == 'NN_basic':
            self.initialize_nn_population(state_dimension, action_dimension, seed)
        else:
            raise RuntimeError('DENSER model type not selected or incorrect in config file.')

        if global_weights is not None:
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
        self.generation_counter = 0

    def generate_random_genotype(self, grammar, symbol="Network"):
        """
        Recursively generate a random genotype (string) from a given grammar.
        """
        if symbol not in grammar:
            return symbol
        choices = np.array(grammar[symbol], dtype=object)
        rule = self.rng.choice(choices)
        return " ".join(self.generate_random_genotype(grammar, sym) for sym in rule)
    
    def initialize_optimizer_population(self, action_dimension, seed):
        """
        Initialize population for an "optimizer" model.
        """
        for i in range(self.population_size):
            # Use a random genotype (even if structure is not used in this branch).
            genotype = self.generate_random_genotype(GRAMMAR, "Network")
            individual = {
                'genotype': genotype,
                'structure': None,
                'weights': self.rng.random(action_dimension),
                'fitness': float('-inf')
            }
            self.population.append(individual)

    def initialize_nn_population(self, state_dimension, action_dimension, seed):
        """
        Initialize population for a neural network model with deliberate structural diversity.
        """
        for i in range(self.population_size):
            # Generate initial genotypes with varying complexity
            if i < self.population_size // 3:
                layer_count = self.rng.integers(1, 3)
                genotype_parts = []
                prev_units = state_dimension
                for _ in range(layer_count):
                    units = self.rng.choice(["16", "32", "64"])
                    act = self.rng.choice(["relu", "tanh", "sigmoid"])
                    genotype_parts.append(f"dense ( {units} ) {act}")
                genotype = " ".join(genotype_parts)
            elif i < 2 * (self.population_size // 3):
                layer_count = self.rng.integers(2, 5)
                genotype_parts = []
                for _ in range(layer_count):
                    units = self.rng.choice(["32", "64", "128"])
                    act = self.rng.choice(["relu", "tanh", "sigmoid"])
                    genotype_parts.append(f"dense ( {units} ) {act}")
                genotype = " ".join(genotype_parts)
            else:
                layer_count = self.rng.integers(3, 7)
                genotype_parts = []
                for _ in range(layer_count):
                    units = self.rng.choice(["64", "128", "256"])
                    act = self.rng.choice(["relu", "tanh", "sigmoid"])
                    genotype_parts.append(f"dense ( {units} ) {act}")
                genotype = " ".join(genotype_parts)
            
            structure = decode_genotype(genotype, state_dimension, action_dimension)
            structure = structure.to(self.device)
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
            for i in range(self.population_size):
                tokens = self.best_individual['genotype'].split()
                if tokens:
                    idx = self.rng.integers(0, len(tokens))
                    if tokens[idx] in GRAMMAR:
                        choices = np.array(GRAMMAR[tokens[idx]], dtype=object)
                        replacement = self.rng.choice(self.rng.choice(choices))
                        tokens[idx] = replacement
                mutated_genotype = " ".join(tokens)
                new_structure = decode_genotype(mutated_genotype, self.in_size, self.out_size)
                new_structure = new_structure.to(self.device)
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
        # old_best_fitness = self.best_individual['fitness']
        # best_idx = -1
        best_gen_fitness = rewards[0]
        
        for i, reward in enumerate(rewards):
            self.population[i]['fitness'] = reward
            if reward > self.best_individual['fitness']:
                # best_idx = i

                # structural_change = True
                # if self.best_individual['genotype'] is not None:
                #     structural_change = self.best_individual['genotype'] != self.population[i]['genotype']

                self.best_individual = {
                    'genotype': self.population[i]['genotype'],
                    'structure': self.population[i]['structure'],
                    'fitness': reward
                }
            
        self.fitness_history.append(max(rewards))
        
        self.evolve_population()
        self.generation_counter += 1


    def evolve_population(self):
        """
        Evolve the population using more aggressive exploration strategies.
        """
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        elite_size = max(1, int(0.1 * self.population_size))
        new_population = sorted_population[:elite_size]

        for _ in range(2):
            genotype = self.generate_random_genotype(GRAMMAR, "Network")
            structure = decode_genotype(genotype, self.in_size, self.out_size).to(self.device)
            new_individual = {
                'genotype': genotype,
                'structure': structure,
                'fitness': float('-inf')
            }
            new_population.append(new_individual)
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(sorted_population)
            
            if self.rng.random() < 0.3:
                parent2 = sorted_population[self.rng.integers(0, len(sorted_population))]
            else:
                parent2 = self.tournament_selection(sorted_population)
            
            if self.rng.random() < 0.8:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            if self.rng.random() < 0.6:
                child = self.mutate(child)
            
            # # Verify the child is actually different from parents
            # if (child['genotype'] != parent1['genotype'] and 
            #     child['genotype'] != parent2['genotype']):
            new_population.append(child)
        
        # Ensure we have the right population size even if some children were rejected
        while len(new_population) < self.population_size:
            print(f'Missing population entering in second while loop')
            genotype = self.generate_random_genotype(GRAMMAR, "Network")
            structure = decode_genotype(genotype, self.in_size, self.out_size).to(self.device)
            new_individual = {
                'genotype': genotype,
                'structure': structure,
                'fitness': float('-inf')
            }
            new_population.append(new_individual)
        
        # Print population diversity statistics for debugging
        genotypes = set(ind['genotype'] for ind in new_population)
        
        self.population = new_population

    def tournament_selection(self, population, tournament_size=3):
        """
        Selects an individual from the population using tournament selection.
        """
        tournament = self.rng.choice(population, min(tournament_size, len(population)), replace=False)
        winner = max(tournament, key=lambda x: x['fitness'])
        return winner.copy()

    def crossover(self, parent1, parent2):
        """
        Performs a more grammar-aware crossover on the genotype strings of two parents.
        Attempts to swap complete layer definitions when possible.
        """
        tokens1 = parent1['genotype'].split()
        tokens2 = parent2['genotype'].split()
        
        if len(tokens1) < 5 or len(tokens2) < 5:
            return {
                'genotype': parent1['genotype'],
                'structure': decode_genotype(parent1['genotype'], self.in_size, self.out_size).to(self.device),
                'fitness': float('-inf')
            }
        
        # Try to find layer boundaries for more meaningful crossover
        layers1 = []
        start_idx = 0
        for i in range(len(tokens1)-4):
            if i >= start_idx and tokens1[i] == "dense" and tokens1[i+1] == "(" and tokens1[i+3] == ")":
                layers1.append((i, i+5))
                start_idx = i+5
        
        layers2 = []
        start_idx = 0
        for i in range(len(tokens2)-4):
            if i >= start_idx and tokens2[i] == "dense" and tokens2[i+1] == "(" and tokens2[i+3] == ")":
                layers2.append((i, i+5))
                start_idx = i+5
        
        
        if layers1 and layers2:
            l1_idx = self.rng.choice(range(len(layers1)))
            l2_idx = self.rng.choice(range(len(layers2)))
                        
            # Extract pre- and post-layer segments
            if l1_idx == 0:
                pre1 = []
            else:
                pre1 = tokens1[:layers1[l1_idx-1][1]]
            
            if l2_idx == len(layers2)-1:
                post2 = []
            else:
                post2 = tokens2[layers2[l2_idx+1][0]:]
            
            layer2 = tokens2[layers2[l2_idx][0]:layers2[l2_idx][1]]
            
            child_tokens = pre1 + layer2 + post2
        else:
            cp1 = self.rng.integers(1, len(tokens1))
            cp2 = self.rng.integers(1, len(tokens2))
            child_tokens = tokens1[:cp1] + tokens2[cp2:]
        
        child_genotype = " ".join(child_tokens)
        
        try:
            child_structure = decode_genotype(child_genotype, self.in_size, self.out_size)
            child_structure = child_structure.to(self.device)
            
            if parent1['structure'] is not None:
                self.transfer_compatible_weights(parent1['structure'], child_structure)
            
            return {
                'genotype': child_genotype,
                'structure': child_structure,
                'fitness': float('-inf')
            }
        except Exception as e:
            # Create a proper deep copy of parent1 instead of a reference
            return {
                'genotype': parent1['genotype'],
                'structure': decode_genotype(parent1['genotype'], self.in_size, self.out_size).to(self.device),
                'fitness': float('-inf')
            }

    def mutate(self, individual):
        """
        Much more aggressive mutation operator with higher chance of structural changes.
        """
        tokens = individual['genotype'].split()
        original_genotype = individual['genotype']
        
        # Higher chance of structural mutation
        mutation_type = self.rng.choice(["nonterminal", "terminal", "structure", "structure", "structure"])
        
        if mutation_type == "structure":
            # STRUCTURAL CHANGES (adding/removing/replacing layers)
            change_type = self.rng.choice(["add", "remove", "replace"])
            
            if change_type == "add" and len(tokens) >= 5:
                # Add a new layer at a random position
                layer_indices = []
                for i in range(len(tokens)-4):
                    if tokens[i] == "dense" and tokens[i+1] == "(" and tokens[i+3] == ")":
                        layer_indices.append(i)
                
                if layer_indices:
                    # Insert after a random existing layer
                    pos = layer_indices[self.rng.choice(len(layer_indices))] + 5
                    units = self.rng.choice(["32", "64", "128", "256"])
                    act = self.rng.choice(["relu", "tanh", "sigmoid"])
                    new_layer = ["dense", "(", units, ")", act]
                    tokens = tokens[:pos] + new_layer + tokens[pos:]
            
            elif change_type == "remove" and len(tokens) > 10:  # Ensure we have at least 2 layers
                # Remove a random layer
                layer_indices = []
                for i in range(len(tokens)-4):
                    if tokens[i] == "dense" and tokens[i+1] == "(" and tokens[i+3] == ")":
                        layer_indices.append(i)
                
                if len(layer_indices) > 1:  # Don't remove if only one layer
                    idx_to_remove = layer_indices[self.rng.choice(len(layer_indices))]
                    tokens = tokens[:idx_to_remove] + tokens[idx_to_remove+5:]
            
            elif change_type == "replace" and len(tokens) >= 5:
                # Replace a layer with a new one
                layer_indices = []
                for i in range(len(tokens)-4):
                    if tokens[i] == "dense" and tokens[i+1] == "(" and tokens[i+3] == ")":
                        layer_indices.append(i)
                
                if layer_indices:
                    idx_to_replace = layer_indices[self.rng.choice(len(layer_indices))]
                    units = self.rng.choice(["32", "64", "128", "256"])
                    act = self.rng.choice(["relu", "tanh", "sigmoid"])
                    tokens[idx_to_replace+2] = units  # Change units
                    tokens[idx_to_replace+4] = act    # Change activation
        
        elif mutation_type == "terminal" and len(tokens) > 0:
            # Mutate unit values or activation functions
            unit_indices = [i for i, tok in enumerate(tokens) if tok in ["4", "8", "16", "32", "64", "128", "256"]]
            act_indices = [i for i, tok in enumerate(tokens) if tok in ["relu", "tanh", "sigmoid", "linear"]]
            
            if unit_indices:
                # Mutate multiple units
                for _ in range(min(2, len(unit_indices))):  # Mutate up to 2 unit values
                    idx = unit_indices[self.rng.choice(len(unit_indices))]
                    units = self.rng.choice(GRAMMAR["units"])[0]
                    tokens[idx] = units
            
            if act_indices:
                # Mutate activation functions
                for _ in range(min(2, len(act_indices))):  # Mutate up to 2 activations
                    idx = act_indices[self.rng.choice(len(act_indices))]
                    activation = self.rng.choice(GRAMMAR["activation"])[0]
                    tokens[idx] = activation
        
        elif mutation_type == "nonterminal":
            # Basic symbol mutation (less important now)
            indices = [i for i, tok in enumerate(tokens) if tok in GRAMMAR]
            if indices:
                idx = self.rng.choice(indices)
                choices = np.array(GRAMMAR[tokens[idx]], dtype=object)
                replacement = self.rng.choice(self.rng.choice(choices))
                tokens[idx] = replacement
        
        new_genotype = " ".join(tokens)
        
        # If no actual change happened, force a change
        if new_genotype == original_genotype:
            # Force a simple change - change a unit value if possible
            unit_indices = [i for i, tok in enumerate(tokens) if tok in ["4", "8", "16", "32", "64", "128", "256"]]
            if unit_indices:
                idx = unit_indices[self.rng.choice(len(unit_indices))]
                current = tokens[idx]
                # Make sure we choose a different value
                choices = [c for c in ["4", "8", "16", "32", "64", "128", "256"] if c != current]
                tokens[idx] = self.rng.choice(choices)
                new_genotype = " ".join(tokens)
        
        try:
            new_structure = decode_genotype(new_genotype, self.in_size, self.out_size)
            new_structure = new_structure.to(self.device)
            
            # Transfer compatible weights from the original structure
            self.transfer_compatible_weights(individual['structure'], new_structure)
            
            changed = new_genotype != original_genotype
            
            return {
                'genotype': new_genotype,
                'structure': new_structure,
                'fitness': float('-inf')
            }
        except Exception as e:
            print(f"Error in mutation: {e}")
            # Fall back to original
            return {
                'genotype': individual['genotype'],
                'structure': decode_genotype(individual['genotype'], self.in_size, self.out_size).to(self.device),
                'fitness': float('-inf')
            }

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

    def evaluate_individual(self, individual, train_loader, val_loader):
        """
        Evaluates an individual's phenotype (the decoded network) by training it for a few epochs
        and computing the fitness (here, negative total validation loss).
        """
        model = individual['structure'].to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)  # Choose an appropriate loss based on your task.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3).to(self.device)
        epochs = 3  # For a proxy evaluation—full training would take longer.
        
        # Train phase (simplified)
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                
        # Validation phase
        model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                batch_count += 1
        avg_val_loss = total_loss / batch_count if batch_count > 0 else total_loss
        
        # Define fitness as negative validation loss so that lower loss gives higher fitness.
        fitness = -avg_val_loss
        individual['fitness'] = fitness
        return fitness

    # (Optional) A method to run the denser model for an input state.
    def run(self, state, individual_index=0):
        """
        Runs the network of a given individual on the provided state.
        """
        if self.model_type == 'NN_basic':
            individual = self.population[individual_index]
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = individual['structure'](state_tensor)
            return output
        else:
            # For 'optimizer' branch, simply return weights (dummy implementation)
            individual = self.population[individual_index]
            return individual['weights']

    # Add this new method to transfer weights between compatible layers
    def transfer_compatible_weights(self, source_model, target_model):
        """
        Transfer weights from source model to target model where layer shapes match.
        This allows preserving learned weights during structural evolution.
        """
        # Get state dictionaries
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        # Transfer matching weights
        transferred_dict = {}
        for target_key in target_dict:
            if target_key in source_dict:
                source_shape = source_dict[target_key].shape
                target_shape = target_dict[target_key].shape
                
                if source_shape == target_shape:
                    # Shapes match, can transfer directly
                    transferred_dict[target_key] = source_dict[target_key]
                elif len(source_shape) == len(target_shape) == 2:
                    # For matrices (Linear layer weights), transfer the overlapping parts
                    rows = min(source_shape[0], target_shape[0])
                    cols = min(source_shape[1], target_shape[1])
                    transferred_dict[target_key] = target_dict[target_key]  # Start with random weights
                    transferred_dict[target_key][:rows, :cols] = source_dict[target_key][:rows, :cols]
                else:
                    # Shapes incompatible, keep random initialization
                    transferred_dict[target_key] = target_dict[target_key]
            else:
                # Key not in source, keep random initialization
                transferred_dict[target_key] = target_dict[target_key]
        
        # Load the transferred weights
        target_model.load_state_dict(transferred_dict)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DenserAgent(state_dim, action_dim, num_cars, seed, agent_index, global_weights, experiment_number, device)
    
    # Example: Retrieve the solutions (genotypes) of the current population.
    solutions = agent.get_solutions()
    
    # To evolve, evaluate your individuals with your training and validation loaders,
    # call agent.tell() with the fitness rewards (list of rewards per individual).
    # For demonstration, we can simulate rewards:
    # fake_rewards = [random.uniform(-10, 0) for _ in range(agent.population_size)]
    rng = np.random.default_rng(seed)
    fake_rewards = rng.uniform(-10, 0, size=agent.population_size).tolist()
    agent.tell(fake_rewards)
    
    # Retrieve best solution genotype and its weights (state_dict if NN model).
    best_solution = agent.get_best_solutions()
    
    # Save the current agent state
    agent.save_model("denser_agent_state.pkl")
