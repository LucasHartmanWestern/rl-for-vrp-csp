# Adapted by Santiago August 9, 2024
import os
import time
import copy
import torch
import numpy as np
import random

from data_loader import load_config_file
from agents.denser_agent import DenserAgent
from evaluation import evaluate
from merl_env._pathfinding import haversine

def train_denser(ev_info, 
                 metrics_base_path,
                 experiment_number,
                 chargers, environment,
                 routes, date,
                 action_dim,
                 global_weights,
                 aggregation_num,
                 zone_index,
                 seed,
                 main_seed,
                 device,
                 agent_by_zone,
                 variant,
                 args,
                 fixed_attributes,
                 verbose,
                 display_training_times=False,
                 dtype=torch.float32,
                 save_offline_data=False,
                 train_model=True,
                 old_buffers=None):
    """
    Trains decision-making agents using the DENSER algorithm with a grammar-based representation.
    This updated version calls the agents’ decoded PyTorch models (structure) to get outputs.
    """
    start_time = time.time()  # Start timing the training process
    avg_rewards = []          # List to store average rewards per generation

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Extract unique charger configurations and prepare environment-specific variables
    unique_chargers = np.unique(
        np.array(list(map(tuple, chargers.reshape(-1, 3))),
                 dtype=[('id', int), ('lat', float), ('lon', float)])
    )
    state_dimension = (environment.num_chargers * 3 * 2) + 6  # Calculate state dimension
    model_indices = environment.info['model_indices']

    # Load NN hyperparameters from configuration
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eps_per_save = int(nn_c['eps_per_save'])
    num_episodes = nn_c['num_episodes'] if train_model else 1

    num_cars = environment.num_cars
    num_agents = 1 if agent_by_zone else num_cars  # Determine number of agents

    # Initialize DENSER agents
    denser_agents_list = []
    for agent_idx in range(num_agents):
        initial_weights = None
        if global_weights is not None:
            if agent_by_zone:
                initial_weights = global_weights[zone_index]
            else:
                initial_weights = global_weights[zone_index][model_indices[agent_idx]]
        agent = DenserAgent(state_dimension, action_dim, num_cars, seed, agent_idx, initial_weights, experiment_number)
        denser_agents_list.append(agent)

    # Storage for average outputs (if needed)
    avg_output_values = np.zeros((denser_agents_list[0].max_generation, action_dim))
    best_avg = float('inf')
    best_paths = None
    metrics = []
    trained = False

    # Reset environment for new training episode
    environment.reset_episode(chargers, routes, unique_chargers)
    denser_info = denser_agents_list[0]
    population_size = denser_info.population_size

    environment.cma_store()

    fitnesses = np.empty((population_size, num_agents))

    # Evolution loop over generations
    for generation in range(denser_info.max_generation):
        # Optionally restart evolution if a threshold is reached
        thresh_limit = 2000
        max_limit = (generation % thresh_limit + 1) * population_size * action_dim
        if max_limit >= 2000 * 20 * 3:
            for agent in denser_agents_list:
                agent.denser_restart()

        sim_done = False
        time_start_paths = time.time()
        timestep_counter = 0

        # --- Evaluate each candidate in the current population ---
        while not sim_done:
            environment.init_routing()
            reward_timestep = 0
            
            # For each candidate solution in population:
            for pop_idx in range(population_size):
                environment.cma_copy_store()  # Restore environment state
                # For each car in the simulation:
                for car_idx in range(num_cars):
                    state = environment.reset_agent(car_idx, timestep_counter)
                    agent_idx = 0 if agent_by_zone else car_idx
                    agent = denser_agents_list[agent_idx]
                    # Get the candidate individual's structure
                    candidate = agent.population[pop_idx]
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    # Forward pass through candidate's network
                    car_route = candidate['structure'](state_tensor).detach().cpu().numpy()
                    environment.generate_paths(car_route, None, agent_idx)

                # Obtain rewards from environment
                sim_done = environment.simulate_routes(timestep_counter)
                _, _, _, _, rewards_pop, _ = environment.get_results()

                # Compute fitness
                if agent_by_zone:
                    fitnesses[pop_idx] = -1 * rewards_pop.sum(axis=0).mean()
                elif 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']:
                    fitnesses[pop_idx] = -1 * rewards_pop.sum(axis=0).mean() * num_cars
                else:
                    fitnesses[pop_idx] = -1 * rewards_pop.sum(axis=0)

        # Update each agent with the fitness values from their evaluated population
        for agent_idx, agent in enumerate(denser_agents_list):
            agent.tell(fitnesses[:, agent_idx].flatten())

        # --- Evaluate the best candidate in each agent using the updated population ---
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False
        timestep_counter = 0
        rewards = []
        while not sim_done:
            environment.init_routing()
            start_time_step = time.time()
            for car_idx in range(num_cars):
                state = environment.reset_agent(car_idx, timestep_counter)
                agent_idx = 0 if agent_by_zone else car_idx
                agent = denser_agents_list[agent_idx]
                # Retrieve best individual's decoded network model
                best_model = agent.best_individual['structure']
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                car_route = best_model(state_tensor).detach().cpu().numpy()
                environment.generate_paths(car_route, None, agent_idx)
                
            sim_done = environment.simulate_routes(timestep_counter)
            sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards, arrived_at_final  = environment.get_results()  # Get the resulting rewards

            if timestep_counter == 0:
                episode_rewards = np.expand_dims(time_step_rewards, axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards, time_step_rewards))
            # Option to average timestep rewards for training
            if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']:
                avg_reward = time_step_rewards.sum(axis=0).mean()
                time_step_rewards_avg = [avg_reward for _ in time_step_rewards]
                rewards.extend(time_step_rewards_avg)
            else:
                rewards.extend(time_step_rewards)

            time_step_time = time.time() - start_time_step

            metric = {
                "zone": zone_index,
                "episode": generation,
                "timestep": timestep_counter,
                "aggregation": aggregation_num,
                "paths": sim_path_results,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": time_step_rewards,
                "best_reward": best_avg,
                "timestep_real_world_time": time_step_time,
                "done": sim_done
            }
            metrics.append(metric)
            timestep_counter += 1

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed))
        if verbose:
            elapsed_time = time.time() - start_time
            to_print = f'(Aggregation: {aggregation_num+1} Zone: {zone_index+1} Generation: {generation+1}/{denser_info.max_generation}) - avg reward {avg_rewards[-1][0]:.3f}'
            print_log(to_print, date, elapsed_time)

        # Save metrics periodically
        if ((generation + 1) % eps_per_save == 0 and generation > 0 and train_model) or (generation == denser_info.max_generation - 2):
            metrics_path = f"{metrics_base_path}/{'eval' if args.eval else 'train'}"
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            evaluate(ev_info, metrics, seed, date, verbose, 'save', num_episodes, f"{metrics_path}/metrics", True)
            metrics = []

        if avg_reward > best_avg:
            best_avg = avg_reward
            if verbose:
                to_print = f'Zone: {zone_index+1} Gen: {generation+1}/{denser_info.max_generation} - New Best: {best_avg:.3f}'
                print_log(to_print, date, None)

        # For logging, store the average of each generation's best individual's "output"
        # Here we extract best individual's output by running its structure on a dummy state
        dummy_state = torch.zeros(state_dimension, dtype=torch.float32).to(device)
        generation_output = []
        for agent in denser_agents_list:
            output = agent.best_individual['structure'](dummy_state).detach().cpu().numpy()
            generation_output.append(output)
        avg_output_values[generation] = np.mean(generation_output, axis=0)

    # End of population evolution
    sim_path_results, sim_traffic, sim_battery_levels, sim_distances, rewards, arrived_at_final = environment.get_results()
    print(f'Rewards for population evolution: {np.mean(rewards):.3f} after {denser_info.max_generation} generations')

    # Save the trained models to disk
    folder_path = 'saved_networks'
    fname = f'{folder_path}/denser_model_{main_seed}_z{zone_index}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for idx, agent in enumerate(denser_agents_list):
        agent.save_model(f'{fname}_agent{idx}.pkl')

    elapsed_time = time.time() - start_time
    weights_list = [agent.get_weights() for agent in denser_agents_list]
    return weights_list, avg_rewards, avg_output_values, metrics, None

def print_log(label, date, et):
    """
    Prints log messages to the console and writes to a log file.
    """
    if et is not None:
        to_print = f"{label}\t - et {str(int(et // 3600)).zfill(2)}:{str(int(et // 60) % 60).zfill(2)}:{str(int(et % 60)).zfill(2)}.{int((et * 1000) % 1000)}"
    else:
        to_print = label
    log_fname = f'logs/{date}-training_logs.txt'
    if not os.path.exists(os.path.dirname(log_fname)):
        os.makedirs(os.path.dirname(log_fname))
    with open(log_fname, 'a') as file:
        print(to_print, file=file)
    print(to_print)
