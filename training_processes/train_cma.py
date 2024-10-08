# Adapted by Santiago August 9, 2024
import os
import time
import copy
import torch
import numpy as np

from agents.cma_agent import CMAAgent


from merl_env._pathfinding import haversine

def train_cma(experiment_number, chargers, environment, routes, date, action_dim, global_weights, aggregation_num,
              zone_index, seed, main_seed, device, agent_by_zone, fixed_attributes, verbose,
              display_training_times=False, dtype=torch.float32, save_offline_data=False, train_model=True):
    """
    Trains decision-making agents using the Covariance Matrix Adaptation (CMA) algorithm.

    Parameters:
        chargers (array): Array containing charger configurations.
        environment (Environment): The simulation environment where agents operate.
        routes (array): Array of available routes for agents to navigate.
        date (str): Date string for logging purposes.
        action_dim (int): Dimensionality of the action space for the agents.
        global_weights (array): Pre-trained weights for initializing the agents.
        aggregation_num (int): The current aggregation number in the experiment.
        zone_index (int): Index of the zone being trained in the simulation.
        seed (int): Random seed for reproducibility in agent training.
        main_seed (int): Seed used for the overall experiment setup.
        device (torch.device): Device (CPU/GPU) where the models are trained.
        agent_by_zone (bool): Flag indicating if agents are assigned by zone or by car.
        fixed_attributes (dict): Fixed attributes for initializing the environment.
        verbose (bool): Flag to control the verbosity of output messages.
        display_training_times (bool): Flag to display the total training time after execution.
        dtype (torch.dtype): Data type for the tensors used in training.
        save_offline_data (bool): Flag indicating whether to save data for offline analysis.

    Returns:
        weights_list (list): List containing the trained weights for each agent.
        avg_rewards (list): List of average rewards obtained per generation.
        avg_output_values (ndarray): Array of average output values across generations.
        metrics (list): List of performance metrics gathered during training.
        trajectories (list): Saved trajectories for further analysis.
    """
    start_time = time.time()  # Start timing the training process
    avg_rewards = []  # List to store average rewards for each generation

    # Set seeds for reproducibility, ensuring consistent results across runs
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)

    # Extract unique charger configurations and prepare environment-specific variables
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))
    state_dimension = (environment.num_chargers * 3 * 2) + 4  # Calculate the state dimension
    model_indices = environment.info['model_indices']

    num_cars = environment.num_cars
    num_agents = 1 if agent_by_zone else num_cars  # Determine number of agents based on assignment mode

    # Initialize CMA agents
    cma_agents_list = []

    for agent_idx in range(num_agents):
        initial_weights = None
        if global_weights is not None:
            # Assign weights based on whether agents are assigned by zone or by car
            if agent_by_zone:
                initial_weights = global_weights[zone_index]
            else:
                initial_weights = global_weights[zone_index][model_indices[agent_idx]]

        # Create a new CMA agent with the provided parameters and initial weights
        cma_agent = CMAAgent(state_dimension, action_dim, num_cars, seed, agent_idx, initial_weights, experiment_number)
        cma_agents_list.append(cma_agent)

    avg_output_values = np.zeros((cma_agents_list[0].max_generation, action_dim))  # Initialize output values storage
    best_avg = float('-inf')  # Track the best average reward encountered
    best_paths = None  # Store the best paths observed
    metrics = []  # Initialize metrics list to track performance
    trained = False  # Track whether training occurred

    # Reset the environment for a new training episode
    environment.reset_episode(chargers, routes, unique_chargers)

    cma_info = cma_agents_list[0]  # Retrieve information from the first CMA agent
    population_size = cma_info.population_size  # Size of the population for evolution

    generation_weights = np.empty((num_agents, action_dim))  # Storage for weights per generation

    # Save the current state of the environment for later restoration during evolution
    environment.cma_store()

    # Initialize matrices for storing solutions and fitness values during evolution
    matrix_solutions = np.zeros((population_size, num_agents, cma_info.out_size))
    fitnesses = np.empty((population_size, num_agents))


    # Evolution process: Loop over generations to evolve the population
    for generation in range(cma_info.max_generation):
        # Generate solutions for each agent in the population
        for agent_idx, agent in enumerate(cma_agents_list):
            matrix_solutions[:, agent_idx, :] = agent.get_solutions()

        sim_done = False
        time_start_paths = time.time()
        timestep_counter = 0

        while not sim_done:  # Keep going until every EV reaches its destination

            environment.init_routing()
            reward_timestep = 0

            start_time_step = time.time()
    
            # Evaluate each individual in the population
            for pop_idx in range(population_size):
                environment.cma_copy_store()  # Restore environment to its stored state
    
                # Simulate the environment for each car
                for car_idx in range(num_cars):

                    start_time_step = time.time()
                    state = environment.reset_agent(car_idx)  # Reset environment for the car #MAYBE A PROBLEM HERE!
                    agent_idx = 0 if agent_by_zone else car_idx  # Determine the agent to use
                    agent = cma_agents_list[agent_idx]
                    weights = matrix_solutions[pop_idx, agent_idx, :]  # Get the agent's weights
                    car_route = agent.model(state, weights)  # Get the route from the agent's model
                    environment.generate_paths(car_route, None, agent_idx)  # Stack the generated paths in the environment
    
                # Once all cars have routes, simulate routes in environment and get results
                sim_done = environment.simulate_routes()
                _, _, _, _, rewards = environment.get_results()  # Retrieve rewards
                reward_timestep += rewards

                if agent_by_zone:
                    fitnesses[pop_idx] = -1 * reward_timestep.mean()
                else:
                    fitnesses[pop_idx] =  np.array(-1 * reward_timestep.mean())*num_cars

        # Update the agents based on the fitness of the solutions
        for agent_idx, agent in enumerate(cma_agents_list):
            agent.es.tell(matrix_solutions[:, agent_idx, :], fitnesses[:, agent_idx].flatten())

        # Get rewards with best solutions after evolving population
        environment.reset_episode(chargers, routes, unique_chargers)

        sim_done = False
        timestep_counter = 0

        rewards = []
        while not sim_done:  # Keep going until every EV reaches its destination
            # environment.cma_copy_store()  # Restore environment to its stored state
            environment.init_routing()
            
            for car_idx in range(num_cars):
                state = environment.reset_agent(car_idx)
                agent_idx = 0 if agent_by_zone else car_idx  # Determine the agent to use
                agent = cma_agents_list[agent_idx]
                weights = agent.get_best_solutions()  # Get the best solutions
                car_route = agent.model(state, weights)  # Generate paths based on best weights
                environment.generate_paths(car_route, None, agent_idx)
                generation_weights[agent_idx] = weights  # Store the best weights for this generation
    
            sim_done = environment.simulate_routes()  # Simulate the environment with the best solutions
            sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = environment.get_results()  # Get the resulting rewards

            if timestep_counter == 0:
                episode_rewards = np.expand_dims(time_step_rewards,axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards,time_step_rewards))

            rewards.extend(episode_rewards.sum(axis=0))
            # rewards.append(episode_rewards.sum(axis=0))
            

            time_step_time = time.time() - start_time_step

            # Used to evaluate simulation
            # metric = {
            #     "zone": zone_index,
            #     "episode": generation,
            #     "timestep": timestep_counter,
            #     "aggregation": aggregation_num,
            #     "paths": sim_path_results,
            #     "traffic": sim_traffic,
            #     "batteries": sim_battery_levels,
            #     "distances": sim_distances,
            #     "rewards": rewards,
            #     "timestep_real_world_time": time_step_time,
            #     "done": sim_done
            # }
            # metrics.append(metric)
            timestep_counter += 1

        avg_reward = episode_rewards.sum(axis=0).mean()

        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed))  # Store the average reward
        # Print information at the log and command line
        if verbose:
            elapsed_time = time.time() - start_time
            to_print = f'(Aggregation: {aggregation_num + 1} Zone: {zone_index + 1} ' +\
                        f'Generation: {generation + 1}/{cma_info.max_generation}) - avg reward {avg_rewards[-1][0]:.3f}'
            print_log(to_print, date, elapsed_time)

        # Compare each generation's best reward and save scores and actions
        if avg_reward > best_avg:
            best_avg = avg_reward
            if verbose:
                to_print = (f' Zone: {zone_index + 1} Gen: {generation + 1}/{cma_info.max_generation} - New Best: {best_avg:.3f}')
                print_log(to_print, date, None)
                # print(f'Zone: {zone_index + 1} - New Best: {best_avg}')

        # Used to evaluate simulation
        metric = {
            "zone": zone_index,
            "episode": generation,
            "timestep": timestep_counter,
            "aggregation": aggregation_num,
            "paths": sim_path_results,
            "traffic": sim_traffic,
            "batteries": sim_battery_levels,
            "distances": sim_distances,
            "rewards": rewards,
            "rewards_mean": episode_rewards.sum(axis=0).mean(),
            "best_reward": best_avg,
            "done": sim_done
        }
        metrics.append(metric)
        avg_output_values[generation] = generation_weights.mean(axis=0)  # Store the average weights for the generation

    # Population evolution ends

    # Retrieve and print results for the best population after evolution
    sim_path_results, sim_traffic, sim_battery_levels, sim_distances, rewards = environment.get_results()
    print(f'Rewards for population evolution: {rewards.mean():.3f} after {cma_info.max_generation} generations')

    # Save the trained models to disk
    folder_path = 'saved_networks'
    fname = f'{folder_path}/cma_model_{main_seed}_z{zone_index}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, agent in enumerate(cma_agents_list):
        agent.save_model(f'{fname}_agent{idx}.pkl')  # Save each agent's model

    elapsed_time = time.time() - start_time  # Calculate total elapsed time for training
    # if verbose:
    #     to_print = (f' Finish Zone: {zone_index + 1} Best reward: {best_avg:.3f}')
    #     print_log(to_print, date, elapsed_time)
    


    ########### STORE EXPERIENCES ########

    if verbose and trained:
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

        print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

    trajectories = []
    weights_list = [agent.get_weights() for agent in cma_agents_list]
  
    return weights_list, avg_rewards, avg_output_values, metrics, trajectories


def print_log(label, date, et):
    """
    Prints log messages to the console and a file.

    Parameters:
        label (str): The log message.
        date (str): The current date.
        et (float): Elapsed time since the start of training.
    """
    if et != None:
        to_print = f"{label}\t - et {str(int(et // 3600)).zfill(2)}:{str(int(et // 60) % 60).zfill(2)}:{str(int(et % 60)).zfill(2)}.{int((et * 1000) % 1000)}"
    else:
        to_print = label
    with open(f'logs/{date}-training_logs.txt', 'a') as file:
        print(to_print, file=file)
    print(to_print)

    