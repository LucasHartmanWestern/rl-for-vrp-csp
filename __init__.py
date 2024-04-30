from dqn_custom import train_dqn
from ev_simulation_environment import get_charger_data, get_charger_list
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
import numpy as np

mp.set_start_method('spawn', force=True)  # This needs to be done before you create any processes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(thread_num, date):
    ############ Algorithm ############

    algorithm = "DQN" # Currently the only option working

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
    usage_per_min = np.array([env_c['usage_per_min'] for agent in range(env_c['num_of_agents'])])
    max_charge = np.array([env_c['max_charge'] for agent in range(env_c['num_of_agents'])])

    batch_size = int(nn_c['batch_size'] * env_c['num_of_agents'])
    buffer_limit = int(batch_size)

    action_dim = nn_c['action_dim'] * env_c['num_of_chargers']

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
            starting_charge[agent] += 1000 * random.randrange(-1, 1)

        elapsed_time = time.time() - start_time
        ev_info = np.vstack((starting_charge, max_charge, usage_per_min))

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        all_routes = [None for route in env_c['coords']]
        for index, (city_lat, city_long) in enumerate(env_c['coords']):
            all_routes[index] = [get_org_dest_coords((city_lat, city_long), env_c['radius'], seeds + i + index) for i in range(env_c['num_of_agents'])]

        elapsed_time = time.time() - start_time

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        chargers = np.zeros(shape=[len(all_routes), env_c['num_of_agents'], env_c['num_of_chargers'] * 3, 3])

        for route_id,  route in enumerate(all_routes):
            for agent_id, (org_lat, org_long, dest_lat, dest_long) in enumerate(route):
                data = get_charger_data()
                charger_info = np.c_[data['latitude'].to_list(), data['longitude'].to_list()]
                charger_list = get_charger_list(charger_info, org_lat, org_long, dest_lat, dest_long, env_c['num_of_chargers'])
                chargers[route_id][agent_id] = charger_list

        elapsed_time = time.time() - start_time
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        user_input = ""

        while user_input != 'Done':
            if train_model:

                if user_input != "":
                    nn_c['num_episodes'] = int(user_input)
                    start_from_previous_session = True
                    nn_c['epsilon'] = 0.1

                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(f"Training using Deep-Q Learning - Session {session}", file=file)

                print(f"Training using Deep-Q Learning - Session {session}")

                rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
                output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
                global_weights = None

                for aggregate_step in range(nn_c['aggregation_count']):

                    print("Get Manager")

                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(chargers))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(chargers))

                    print("Get Processes")

                    processes = []
                    for ind, charger_list in enumerate(chargers):
                        process = mp.Process(target=train_route, args=(
                            charger_list, ev_info, all_routes[ind], date, action_dim, global_weights, aggregate_step,
                            ind, seeds, thread_num, nn_c['epsilon'], nn_c['epsilon_decay'], nn_c['discount_factor'],
                            nn_c['learning_rate'], nn_c['num_episodes'], batch_size, buffer_limit,
                            env_c['num_of_agents'], env_c['num_of_chargers'], start_from_previous_session, nn_c['layers'],
                            hpp_c['fixed_attributes'], local_weights_list, process_rewards, process_output_values, barrier))
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
                fixed_attributes = hpp_c['fixed_attributes']
                attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

            if save_data:
                # Add this before you save your model
                if not os.path.exists('outputs'):
                    os.makedirs('outputs')

                # TODO - Update this to not use environments

                # for index, env in enumerate(envs):
                #     env.write_path_to_csv(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                #     env.write_chargers_to_csv(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                #     env.write_reward_graph_to_csv(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                #     env.write_charger_traffic_to_csv(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')

            if generate_plots:
                num_of_agents = env_c['num_of_agents']
                num_episodes = nn_c['num_episodes']

                for index, env in enumerate(chargers):
                    route_data = read_csv_data(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    charger_data = read_csv_data(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    reward_data = read_csv_data(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    traffic_data = read_csv_data(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')

                    route_datasets = []
                    if env_c['num_of_agents'] == 1:
                        for id_value, group in route_data.groupby('Episode Num'):
                            route_datasets.append(group)
                    else:
                        for episode_num, episode_group in route_data.groupby('Episode Num'):
                            if episode_num == route_data['Episode Num'].max():
                                for agent_num, agent_group in episode_group.groupby('Agent Num'):
                                    route_datasets.append(agent_group)

                    if (train_model or start_from_previous_session) and num_episodes > 1:
                        generate_average_reward_plot(algorithm, reward_data, session)

                    if num_episodes == 1 and num_of_agents > 1:
                        generate_traffic_plot(traffic_data)

                    origins = [(route[0], route[1]) for route in all_routes[index]]
                    destinations = [(route[2], route[3]) for route in all_routes[index]]
                    generate_interactive_plot(algorithm, session, route_datasets, charger_data, origins, destinations)

            if nn_c['num_episodes'] != 1 and continue_training:
                user_input = input("More Episodes? ")
            else:
                user_input = 'Done'

def train_route(chargers, ev_info, routes, date, action_dim, global_weights, aggregate_step, ind, seeds, thread_num, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes, batch_size, buffer_limit, num_of_agents, num_of_chargers, start_from_previous_session, layers, fixed_attributes, local_weights_list, rewards, output_values, barrier):
    try:
        # Create a deep copy of the environment for this thread
        chargers_copy = copy.deepcopy(chargers)

        local_weights, avg_rewards, avg_output_values = train_dqn(chargers_copy, ev_info, routes, date, action_dim, global_weights, aggregate_step, ind, seeds, thread_num, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes,
                                  batch_size, buffer_limit, num_of_agents, num_of_chargers, start_from_previous_session, layers, fixed_attributes)

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
