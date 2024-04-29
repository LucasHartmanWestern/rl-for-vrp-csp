from dqn_custom import train_dqn
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import *
from geolocation.maps_free import get_org_dest_coords
from training_visualizer import Simulation
from _helpers import load_config_file as load_config
import random
import os
import time
import torch.multiprocessing as mp
from frl_custom import aggregate_weights
import copy
from datetime import datetime

mp.set_start_method('spawn', force=True)  # This needs to be done before you create any processes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(thread_num, date):
    ############ Configuration ############

    train_model = True # Set to false for model evaluation
    start_from_previous_session = False # Set to true if you want to reuse the models from a previous session
    save_data = False # Set to true if you want to save simulation results in a csv file
    generate_plots = False # Set to true if you want to generate plots of the simulation environments
    save_aggregate_rewards = True # Set to true if you want to save the rewards across aggregations
    continue_training = False # Set to true if you want the option to continue training after a full training loop completes


    ############ Initialization ############
    config_fname = 'configs/nnParameters.yaml'
    c = load_config(config_fname)
    env_c = c['environment_settings']
    nn_c  = c['nn_hyperparameters']
    hpp_c = c['hpp_config']
    
    seeds = env_c['seeds'] * thread_num
    starting_charge = [env_c['starting_charge'] for agent in range(env_c['num_of_agents'])]
    # Run multiple training sessions with differing origins and destinations
    for session in range(nn_c['num_training_sesssions']):

        seeds += session

        if session != 0 and train_model:
            start_from_previous_session = True # Always continue from previous training session when running back-to-back sessions

        if env_c['radius'] is None:
            env_c['radius'] = ((1 / (nn_c['num_training_sesssions'] / 14)) * session) + 1

        start_time = time.time()
        for agent in range(env_c['num_of_agents']):
            random.seed(seeds + agent)
            # Random charge between 0.5-x%, where x scales between 1-25% as sessions continue
            if starting_charge is None:
                starting_charge[agent] = random.randrange(500, int(1000 * (((1 / (nn_c['num_training_sesssions'] / 24)) * session) + 1)), 100)
            else:
                starting_charge[agent] += 1000 * random.randrange(-1, 1)
        elapsed_time = time.time() - start_time
        print(f"Get Starting Battery: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        all_routes = [None for route in env_c['coords']]

        for index, (city_lat, city_long) in enumerate(env_c['coords']):
            all_routes[index] = [get_org_dest_coords((city_lat, city_long), env_c['radius'], seeds + i + index) for i in range(env_c['num_of_agents'])]

        elapsed_time = time.time() - start_time
        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        envs = [None for route in all_routes]
        for index,  routes in enumerate(all_routes):
            envs[index] = EVSimEnvironment(nn_c['max_num_timesteps'], nn_c['num_episodes'], env_c['num_of_chargers'], env_c['make'], env_c['model'], starting_charge, env_c['max_charge'], routes, seeds + index)

        elapsed_time = time.time() - start_time
        print(f"Initialize Environment: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        user_input = ""

        while user_input != 'Done':
            if train_model:

                if user_input != "":
                    nn_c['num_episodes'] = int(user_input)
                    start_from_previous_session = True
                    nn_c['epsilon'] = 0.1

                state_dimension = env_c['num_of_chargers'] * 6 + 3   # Traffic level and distance of each station plus total charger num, total distance, and number of EVs
                action_dimension = env_c['num_of_chargers'] * 3      # attributes for each station

                print(f"Training using Deep-Q Learning - Session {session}")

                rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
                output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
                global_weights = None

                for aggregate_step in range(nn_c['aggregation_count']):

                    print("Get Manager")
                    
                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(envs))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(envs))

                    print("Get Processes")
                    
                    processes = []
                    for ind, env in enumerate(envs):
                        process = mp.Process(target=train_route, args=(
                            env, date, global_weights, aggregate_step, ind, seeds, thread_num, nn_c, env_c['num_of_agents'], state_dimension,
                            action_dimension, start_from_previous_session, hpp_c['fixed_attributes'],
                            local_weights_list, process_rewards, process_output_values, barrier))
                        processes.append(process)
                        process.start()

                    print("Join Processes")
                    
                    for process in processes:
                        process.join()

                    print("Join Weights")
                    
                    # Aggregate the weights from all local models
                    global_weights = aggregate_weights(local_weights_list)

                    # Extend the main lists with the contents of the process lists
                    rewards.extend(process_rewards)
                    output_values.extend(process_output_values)

                    with open(f'logs/{date}-training_logs.txt', 'a') as file:
                        print(f"\n\n############ Aggregation Step {aggregate_step} ############\n\n", file=file)

                    print(f"\n\n############ Aggregation Step {aggregate_step} ############\n\n",)

                # Plot the aggregated data
                if save_aggregate_rewards:
                    save_to_csv(rewards, 'outputs/rewards.csv')
                    save_to_csv(output_values, 'outputs/output_values.csv')

                    loaded_rewards = load_from_csv('outputs/rewards.csv')
                    loaded_output_values = load_from_csv('outputs/output_values.csv')

                    plot_aggregate_reward_data(loaded_rewards)
                    plot_aggregate_output_values_per_route(loaded_output_values)

            if hpp_c['fixed_attributes'] != [0, 1] and hpp_c['fixed_attributes'] != [1, 0] and hpp_c['fixed_attributes'] != [0.5, 0.5]:
                attr_label = 'learned'
            else:
                attr_label = '{}_{}'.format(fixed_attributes[0],fixed_attributes[1])

            if save_data:
                # Add this before you save your model
                if not os.path.exists('outputs'):
                    os.makedirs('outputs')

                for index, env in enumerate(envs):
                    out_fname = '{}_{}_{}_{}_{}'.format(env_c['num_of_agents'],nn_c['num_episodes'],seeds,attr_label,index)
                    env.write_path_to_csv(f'outputs/routes_{out_fname}.csv')
                    env.write_chargers_to_csv(f'outputs/chargers_{out_fname}.csv')
                    env.write_reward_graph_to_csv(f'outputs/rewards_{out_fname}.csv')
                    env.write_charger_traffic_to_csv(f'outputs/traffic_{out_fname}.csv')

            if generate_plots:
                for index, env in enumerate(envs):
                    out_fname = '{}_{}_{}_{}_{}'.format(env_c['num_of_agents'],nn_c['num_episodes'],seeds,attr_label,index)
                    route_data = read_csv_data(f'outputs/routes_{out_fname}.csv')
                    charger_data = read_csv_data(f'outputs/chargers_{out_fname}.csv')
                    reward_data = read_csv_data(f'outputs/rewards_{out_fname}.csv')
                    traffic_data = read_csv_data(f'outputs/traffic_{out_fname}.csv')

                    route_datasets = []
                    if env_c['num_of_agents'] == 1:
                        for id_value, group in route_data.groupby('Episode Num'):
                            route_datasets.append(group)
                    else:
                        for episode_num, episode_group in route_data.groupby('Episode Num'):
                            if episode_num == route_data['Episode Num'].max():
                                for agent_num, agent_group in episode_group.groupby('Agent Num'):
                                    route_datasets.append(agent_group)

                    if (train_model or start_from_previous_session) and nn_c['num_episodes'] > 1:
                        generate_average_reward_plot(c['algorithm'], reward_data, session)

                    if nn_c['num_episodes'] == 1 and env_c['num_of_agents'] > 1:
                        generate_traffic_plot(traffic_data)

                    origins = [(route[0], route[1]) for route in all_routes[index]]
                    destinations = [(route[2], route[3]) for route in all_routes[index]]
                    generate_interactive_plot(c['algorithm'], session, route_datasets, charger_data, origins, destinations)

            if nn_c['num_episodes'] != 1 and continue_training:
                user_input = input("More Episodes? ")
            else:
                user_input = 'Done'


def train_route(env, date, global_weights, aggregate_step, ind, seeds, thread_num, nn_c, num_of_agents, state_dimension, action_dimension, start_from_previous_session, fixed_attributes, local_weights_list, rewards, output_values, barrier):
    try:
        # Create a deep copy of the environment for this thread
        env_copy = copy.deepcopy(env)
        
        local_weights, avg_rewards, avg_output_values = train_dqn(env_copy, date, global_weights, aggregate_step, ind, seeds,\
                                                                  thread_num, nn_c, state_dimension, action_dimension, num_of_agents,\
                                                                  start_from_previous_session, fixed_attributes)

        rewards.append(avg_rewards)
        output_values.append(avg_output_values)
        local_weights_list[ind] = local_weights

        print(f"Thread {ind} waiting")

        barrier.wait()  # Wait for all threads to finish before proceeding

    except Exception as e:
        print(f"Error in process {ind} during aggregate step {aggregate_step}: {str(e)}")
        raise


if __name__ == '__main__':

    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d_%H-%M')

    train_rl_vrp_csp(1, date)