import torch
import torch.optim as optim
import numpy as np
from collections import namedtuple
import os
from collections import deque
import time
import copy


from data_loader import load_config_file
from cma_agent import CMAAgent

import collections

# Define the experience tuple
experience = namedtuple("Experience", field_names=["state", "distribution", "reward", "next_state", "done"])

def train_cma(chargers, environment, routes, date, action_dim, global_weights, aggregation_num,\
          zone_index, seed, main_seed, device, agent_by_zone, fixed_attributes, verbose,\
          display_training_times=False, dtype=torch.float32, save_offline_data=False
):

    """
    Trains a Decision maeker for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu. 
                                 device[0] for environment setting, device[1] for model trainning.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        nn_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car


    Returns:
        tuple: A tuple containing:
            - List of trained Q-network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """

    #loading the type of algorithm for the decision maker

    
    
    avg_rewards = []
    
    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)

    # Create variables from environment
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),\
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))
    state_dimension = (environment.num_chargers * 3 * 2) + 4
    model_indices = environment.info['model_indices']

    if agent_by_zone:
        num_agents = 1
        num_cars = environment.num_cars
    else:
        num_agents = environment.num_cars
        num_cars = 1



    # if nn_by_zone:  # Use same NN for each zone
    #     # Initialize networks
    #     q_network, target_q_network = initialize(state_dimension, action_dim, layers, device) 

    #     if global_weights is not None:
    #         q_network.load_state_dict(global_weights[zone_index])
    #         target_q_network.load_state_dict(global_weights[zone_index])

    #     optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

    #     # Store individual networks
    #     q_networks.append(q_network)
    #     target_q_networks.append(target_q_network)
    #     optimizers.append(optimizer)

    # else:  # Assign unique NN for each agent
    #     for agent_ind in range(environment.num_cars):
    #         # Initialize networks
    #         q_network, target_q_network = initialize(state_dimension, action_dim, layers, device)  

    #         if global_weights is not None:
    #             q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
    #             target_q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

    #         optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

    #         # Store individual networks
    #         q_networks.append(q_network)
    #         target_q_networks.append(target_q_network)
    #         optimizers.append(optimizer)

    cma_agents_list = []
    
    for agent_idx in range(num_agents):
        cma_agent = CMAAgent(state_dimension, action_dim, num_cars, seed, agent_idx)
        if global_weights != None:
            cma_agent.load_global_weights(global_weights[zone_index][agent_idx])

        cma_agents_list.append(cma_agent)
        

    trajectories = []
    
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    metrics = []
    trained = False
    avg_output_values = []  # List to store the average values of output neurons for each episode

    environment.reset_episode(chargers, routes, unique_chargers)
    if num_agents == 1:
        cma_agent = cma_agents_list[0]
        action_values, trained = cma_agent.run_scenarios(environment)
        distributions = cma_agent.run_routes(environment)

    elif num_agents > 1:
        cma_info = cma_agents_list[0] 
        scenario_list = [copy.deepcopy(environment) for i in range(cma_info.population_size)]\

        states = []
        actions= []
        fitnesses = np.empty((num_agents, cma_info.population_size,))
        
        for generation in range(cma_info.max_generation):
            agents_solutions = []
            for agent in cma_agents_list: # For each agent
                agents_solutions.append(agent.get_actions())
            
            print(f'agents solutions {len(agents_solutions), len(agents_solutions[0]), len(agents_solutions[0][0])}')

            for scenario_idx, env in enumerate(scenario_list):
                state_scenario = []
                action_scenario= []
                env.idx = scenario_idx
                for car_idx in range(num_cars):
                    state = env.reset_agent(car_idx)
                    weights = agents_solutions[car_idx][scenario_idx]
                    car_route = cma_info.model(state,weights)
                    env.generate_paths(car_route, None)
                    state_scenario.append(state)
                    action_scenario.append(car_route)

                
                env.simulate_routes()
                _,_,_,_, rewards = env.get_results()
                print(f' for scenario {scenario_idx}: actions {action_scenario[-num_cars]}, rewards {rewards}')
                fitnesses[:, scenario_idx] = rewards

                states.append(state_scenario)
                actions.append(action_scenario)

            print(f'fitnesses {fitnesses}')
            for idx, agent in enumerate(cma_agents_list): # For each agent
                agent.es.tell(agents_solutions[idx], fitnesses[idx].flatten())
                agent.es.logger.add()
                agent.es.disp()

                    

    print(f'scenarios best {fitnesses}')   
            


    # for episode in range(num_episodes):  # For each episode


    #     distributions = []
    #     distributions_unmodified = []
    #     states = []
    #     environment.reset_episode(chargers, routes, unique_chargers)
        
    #     if episode > 0:
    #         decision_maker.reset_episode(action_values)
        
    #     time_start_paths = time.time()

    #     action_values = decision_maker.run_scenarios(environment)
    #     distributions = decision_maker.run_routes(environment)


    #     dm_actions, dm_states = decision_maker.get_run_info()
    #     states.append(dm_states)
    #     distributions_unmodified.append(dm_actions)
        
    #     # # Build path for each EV
    #     # for agent_idx in range(environment.num_cars): # For each agent
    #     #     ########### Starting environment rutting
    #     #     state = environment.reset_agent(agent_idx)
    #     #     states.append(state)  # Track states

    #     #     t1 = time.time()

    #     #     action_values = decision_maker.get_actions(state, environment)

    #     #     environment.generate_paths(action_values, fixed_attributes)

    #     #     distributions_unmodified.append(action_values.tolist())
    #     #     distributions.append(action_values.tolist())  # Convert back to list and append
    #     #     # ####### Getting actions from agents
    #     #     # state = torch.tensor(state, dtype=dtype, device=device)  # Convert state to tensor
    #     #     # # action_values = get_actions(state, q_networks, random_threshold, epsilon, episode, agent_idx,\
    #     #     # #                             device, nn_by_zone)  # Get the action values from the agent
    #     #     # action_values = nn_agent.get_actions(state)
            
    #     #     # t2 = time.time()

    #     #     # distribution = action_values.detach().numpy()  # Convert PyTorch tensor to NumPy array
    #     #     # distributions_unmodified.append(distribution.tolist())  # Track outputs before the sigmoid application
    #     #     # distribution = 1 / (1 + np.exp(-distribution))  # Apply sigmoid function to the entire array
    #     #     # distributions.append(distribution.tolist())  # Convert back to list and append

    #     #     # t3 = time.time()

    #     #     # environment.generate_paths(distribution, fixed_attributes)

    #     #     # t4 = time.time()

    #     #     # if agent_idx == 0 and display_training_times:
    #     #     #     print_time("Get actions", (t2 - t1))
    #     #     #     print_time("Get distributions", (t3 - t2))
    #     #     #     print_time("Generate paths in environment", (t4 - t3))


    #     if num_episodes == 1 and fixed_attributes is None:
    #         if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
    #             paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy',\
    #                             allow_pickle=True).tolist()

    #     paths_copy = copy.deepcopy(environment.paths)

    #     # Calculate the average values of the output neurons for this episode
    #     episode_avg_output_values = np.mean(distributions_unmodified, axis=0)
    #     avg_output_values.append((episode_avg_output_values.tolist(), episode, aggregation_num,\
    #                               zone_index, main_seed))

    #     time_end_paths = time.time() - time_start_paths

    #     if display_training_times:
    #         print_time('Get Paths',time_end_paths)

    ########### GET SIMULATION RESULTS ###########

    # Run simulation    
    environment.simulate_routes()
    
    #Get results from environment
    sim_path_results, sim_traffic, sim_battery_levels, sim_distances, rewards = environment.get_results()

    et = time.time() - start_time
    # Used to evaluate simulation
    metric = {
        "zone": zone_index,
        "episode": cma_agent.max_generation,
        "aggregation": aggregation_num,
        "paths": sim_path_results,
        "traffic": sim_traffic,
        "batteries": sim_battery_levels,
        "distances": sim_distances,
        "rewards": rewards
    }
    metrics.append(metric)

    ########### STORE EXPERIENCES ###########

    if verbose and trained:
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

        print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places


    # if episode % 25 == 0 and episode >= buffer_limit:  # Every 25 episodes
    #     if nn_by_zone:
    #         soft_update(target_q_networks[0], q_networks[0])

    #         # Add this before you save your model
    #         if not os.path.exists('saved_networks'):
    #             os.makedirs('saved_networks')

    #         # Save the networks at the end of training
    #         save_model(q_networks[0], f'saved_networks/q_network_{main_seed}_{zone_index}.pth')
    #         save_model(target_q_networks[0], f'saved_networks/target_q_network_{main_seed}_{zone_index}.pth')
    #     else:
    #         for agent_ind in range(num_agents):
    #             soft_update(target_q_networks[agent_ind], q_networks[agent_ind])

    #             # Add this before you save your model
    #             if not os.path.exists('saved_networks'):
    #                 os.makedirs('saved_networks')

    #             # Save the networks at the end of training
    #             save_model(q_networks[agent_ind], f'saved_networks/q_network_{main_seed}_{agent_ind}.pth')
    #             save_model(target_q_networks[agent_ind],\
    #                        f'saved_networks/target_q_network_{main_seed}_{agent_ind}.pth')

    #     # Log every ith episode
    #     if episode % 1 == 0:
    #         avg_reward = 0
    #         for reward in rewards:
    #             avg_reward += reward
    #         avg_reward /= len(rewards)
    #         # Track rewards over aggregation steps
    #         avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

    #         if avg_reward > best_avg:
    #             best_avg = avg_reward
    #             best_paths = paths_copy
    #             if verbose:
    #                 print(f'Zone: {zone_index + 1} - New Best: {best_avg}')

    #         avg_ir = 0
    #         ir_count = 0
    #         for distribution in distributions:
    #             for out in distribution:
    #                 avg_ir += out
    #                 ir_count += 1
    #         avg_ir /= ir_count

    #         elapsed_time = time.time() - start_time

    #         # Open the file in write mode (use 'a' for append mode)
    #         if verbose:
    #             with open(f'logs/{date}-training_logs.txt', 'a') as file:
    #                 print(f"Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}", file=file)

    #             print(f"Aggregation: {aggregation_num + 1} - Zone: {zone_index + 1} - Episode: {episode + 1}/{num_episodes} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}")

    # np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))


    weights_list = [cma_agent.get_weights() for cma_agent in cma_agents_list]
    
    return weights_list, avg_rewards, avg_output_values, metrics, trajectories


def print_time(label,time):
    print(f"{label} - {int(time // 3600)}h, {int((time % 3600) // 60)}m, {int(time % 60)}s")
    