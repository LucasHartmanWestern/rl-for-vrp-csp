import os
import yaml

# Get most recent experiment number
latest_experiment = 40

# Generate experiments as combinations of the following parameters.
#
# There is one experiment for each combination of the following parameters:
# - Model: [DQN, PPO, DQN + ODT, PPO + ODT, ALL + ODT, CMA] (6 models)
# - Seed: [1234, 5555, 2020 (3 seeds)
# - Season: [winter, summer] (2 seasons)
#
# Number of combinations = 6 * 3 * 2 = 36 experiments

algorithm_settings = {
    "algorithm": {
        "permutate": True,
        "options": ["DQN", "PPO", "ODT + DQN", "ODT + PPO", "ODT + ALL", "CMA"]
    },
    "agent_by_zone": False
}

federated_learning_settings = {
    "aggregation_count": 50,
    "city_multiplier": 0.15,
    "zone_multiplier": 0.35,
    "model_multiplier": 0.50,
}

environment_settings = {
    "seed": { # Used for reproducibility
        "permutate": True,
        "options": [1234, 5555, 2020]
    },
    "num_of_cars": 100, # Num of cars in simulation
    "num_of_chargers": 3, # 3x this amount of chargers will be used (for origin, destination, and midpoint)
    "action_dim": 3, # Multiplied by the number of chargers to represent how many output neurons there are for each NN
    "season": {
        "permutate": True,
        "options": ["winter", "summer"]
    },
    "coords": [
        [43.02120034946083, -81.28349087468504], # North London
        [43.004969336049854, -81.18631870502043], # East London
        [42.95923445066671, -81.26016049362336], # South London
        [42.98111190139387, -81.30953935839466], # West London
        # [42.9819404397449, -81.2508736429095] # Central London
    ],
    "radius": 20, # Max radius of the circle containing the entire trip
    "starting_charge": 6000, # 6kw base starting charge
    "models": ['Tesla Model Y', 'BYD Song', 'Tesla Model 3'], # The 3 most popular EV models
    "usage_per_hour": [9840, 8460, 9360], # Average usage per hour for each model in Wh/60 km
    "max_charge": [170000, 90000, 170000], # in Watts
    "step_size": 0.01,  # 60km per hour / 60 minutes per hour = 1 km per minute
    "increase_rate": 12500,
    "max_sim_steps": 50,
    "max_mini_sim_steps": 10,
}

eval_config = {
    "fixed_attributes": [], # Determine impact ratings through training
    "verbose": True, # Log episode-specific data during training
    "display_training_times": False, # Display breakdown of how much each step takes
    "evaluate_on_diff_seed": False, # Set to true to evaluate a model on a different seed than it was trained on
    "evaluate_on_diff_zone": False, # Set to true to evaluate a model on a different zone than it was trained on
    "save_data": True, # Set to true if you want to save simulation results in a csv file
    "save_offline_data": False, # Set true to save trajectories for offline training
    "generate_plots": False, # Set to true if you want to generate plots of the simulation environments
    "save_aggregate_rewards": False, # Set to true if you want to save the rewards across aggregations
    "continue_training": False, # Set to true if you want the option to continue training after a full training loop completes
    "save_path_metrics": '../../../storage_1/metrics/Exp' # Path to store data from experiments
}

nn_hyperparameters = {
    "num_episodes": 200, # Amount of training episodes per session
    "learning_rate": 0.00001, # Rate of change for model parameters
    "epsilon": 1, # Introduce noise during training
    "discount_factor": 0.999, # Present value of future rewards
    "epsilon_decay": 0.999, # Rate of decrease for training noise
    "batch_size": 75, # Amount of experiences to use when training
    "buffer_limit": 150, # Max number of experiences to store in a buffer
    "max_num_timesteps": 300, # Max amount of minutes per agent episode
    "layers": [128, 64, 64], # Neural network hidden layers
    "eps_per_save": 200, # How many episodes can go by before saving
    "log_std_decay_rate": 0.01, # Rate of decay for the log std of the action distribution
    "num_epochs": 10, # Amount of epochs to train the neural network

    # Multipliers to control aggregation
    "city_multiplier": 0.15, # Importance of city-wide average
    "zone_multiplier": 0.35, # Importance of zone-wide average
    "model_multiplier": 0.50 # Importance of model-wide average
}

cma_parameters = {
    "population_dimension": 20,
    "initial_sigma": 0.1,
    "max_generations": 200,
    "model_type": 'optimizer' 
}

odt_hyperparameters = {
    "env": 'hopper',
    "dataset": '[10seeds]-100-3-3-100-North-East',
    "mode": 'normal',
    "K": 10,
    "pct_traj": 1.0,
    "batch_size": 256,
    "model_type": 'dt',
    "embed_dim": 128,
    "n_layer": 3,
    "n_head": 1,
    "activation_function": 'relu',
    "dropout": 0.1,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "warmup_steps": 10000,
    "num_eval_episodes": 1,
    "max_iters": 2,
    "num_steps_per_iter": 300,
    "device": 'cuda:1',
    "log_to_wandb": True,
    "save_model": False,
    "pretrained_model": '/storage_1/epigou_storage/offline_networks/dt_0[10seeds]-100-3-3-100-North-East-620076.pt',
    "stochastic": True,
    "use_entropy": False,
    "use_action_means": True,
    "online_training": True,
    "online_buffer_size": 1001,
    "eval_only": False,
    "remove_pos_embs": False,
    "eval_context": 5,
    "target_entropy": True,
    "stochastic_tanh": True,
    "approximate_entropy_samples": 1000,
    "zone_index": 1 #Second zone
}

# Loop through each experiment permutation
algorithm_options = algorithm_settings["algorithm"]["options"]
seed_options = environment_settings["seed"]["options"]
season_options = environment_settings["season"]["options"]

total_combinations = len(algorithm_options) * len(seed_options) * len(season_options)

for i in range(latest_experiment, latest_experiment + total_combinations):
    # Create YAML file with the proper config for the experiment
    print(f"Creating config for experiment {i}")


    # Define the base directory for experiments
    base_dir = './experiments/Exp_{}'.format(i)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Define the configuration dictionary
    config = {
        "algorithm_settings": algorithm_settings,
        "federated_learning_settings": federated_learning_settings,
        "environment_settings": environment_settings,
        "eval_config": eval_config,
        "nn_hyperparameters": nn_hyperparameters,
        "cma_parameters": cma_parameters,
        "odt_hyperparameters": odt_hyperparameters
    }
    
    ind = i - latest_experiment

    # Update config to get every combination of algorithm, seed, and season
    combination_index = ind % total_combinations

    algorithm_index = combination_index % len(algorithm_options)
    seed_index = (combination_index // len(algorithm_options)) % len(seed_options)
    season_index = (combination_index // (len(algorithm_options) * len(seed_options))) % len(season_options)

    config["algorithm_settings"]["algorithm"] = algorithm_options[algorithm_index]
    config["environment_settings"]["seed"] = seed_options[seed_index]
    config["environment_settings"]["season"] = season_options[season_index]

    # Write the configuration to a YAML file
    with open(os.path.join(base_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    # Create data for experiment description text file
    description = f"""
### Experiment {i}: ###
--------------------------------
Seed: {seed_options[seed_index]}
Model: {algorithm_options[algorithm_index]}
Season: {season_options[season_index]}
Number of cars: {environment_settings['num_of_cars']}
Number of chargers: {environment_settings['num_of_chargers']}
Number of episodes: {nn_hyperparameters['num_episodes']}
Number of aggregations: {federated_learning_settings['aggregation_count']}
"""
    
    # Write the description file
    with open(os.path.join(base_dir, 'description.txt'), 'w') as file:
        file.write(description)

    # Get info to create copy of experiment analysis notebook
    experiment_notebook_path = "./experiments/Experiment_Analysis_Template.ipynb"
    experiment_notebook_name = "Experiment Analysis.ipynb"

    # Create new copy of template
    with open(experiment_notebook_path, 'r') as template_file:
        notebook_content = template_file.read()
    
    with open(os.path.join(base_dir, experiment_notebook_name), 'w') as new_notebook_file:
        new_notebook_file.write(notebook_content)