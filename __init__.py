from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import *
from geolocation.maps_free import get_org_dest_coords
from training_visualizer import Simulation
import random
import os
import time
import threading
from frl_custom import aggregate_weights


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(thread_num):
    print(f'INITIALIZE THREAD {thread_num}')

    ############ Algorithm ############

    algorithm = "DQN" # Currently the only option working

    ############ Configuration ############

    train_model = True # Set to false for model evaluation
    start_from_previous_session = False # Set to true if you want to reuse the models from a previous session
    save_data = False # Set to true if you want to save simulation results in a csv file
    generate_plots = False # Set to true if you want to generate plots of the simulation environments
    plot_aggregate_rewards = True # Set to true if you want to see the rewards across aggregations
    continue_training = False # Set to true if you want the option to continue training after a full training loop completes

    ############ Environment Settings ############

    seeds = 1000 * thread_num # Used for reproducibility
    num_of_agents = 100 # Num of cars in simulation
    num_of_chargers = 5 # 3x this amount of chargers will be used (for origin, destination, and midpoint)
    make = 0 # Not currently used
    model = 0 # Not currently used
    max_charge = 100000 # 100kW

    # Coordinates of city regions in London
    coords = [(43.02120034946083, -81.28349087468504), # North London
              (43.004969336049854, -81.18631870502043), # East London
              (42.95923445066671, -81.26016049362336), # South London
              (42.98111190139387, -81.30953935839466), # West London
              (42.9819404397449, -81.2508736429095) # Central London
    ]

    radius = 20 # Max radius of the circle containing the entire trip
    starting_charge = [6000 for agent in range(num_of_agents)] # 6kw starting charge

    ############ Hyperparameters ############

    aggregation_count = 5 # Amount of aggregation steps for federated learning

    num_training_sesssions = 1 # Depreciated
    num_episodes = 250 # Amount of training episodes per session
    learning_rate = 0.00001 # Rate of change for model parameters
    epsilon = 1 # Introduce noise during training
    discount_factor = 0.999 # Present value of future rewards
    epsilon_decay = 0.995 # Rate of decrease for training noise
    batch_size = 50 * num_of_agents # Amount of experiences to use when training
    max_num_timesteps = 300 # Max amount of minutes per agent episode
    buffer_limit = int(batch_size) # Start training after this many experiences are accumulated
    layers = [64, 64, 64, 128, 64, 64, 64] # Neural network hidden layers

    ############ HPP Settings ############

    # Determine if agent or baseline is used
    # fixed_attributes = [0, 1] # Assign fixed attributes to compare a baseline [Traffic_mult, Distance_mult]
    fixed_attributes = None # Determine impact ratings through training

    ############ Initialization ############

    # Run multiple training sessions with differing origins and destinations
    for session in range(num_training_sesssions):

        seeds += session

        if session != 0 and train_model:
            start_from_previous_session = True # Always continue from previous training session when running back-to-back sessions

        if radius is None:
            radius = ((1 / (num_training_sesssions / 14)) * session) + 1

        start_time = time.time()
        for agent in range(num_of_agents):
            random.seed(seeds + agent)
            # Random charge between 0.5-x%, where x scales between 1-25% as sessions continue
            if starting_charge is None:
                starting_charge[agent] = random.randrange(500, int(1000 * (((1 / (num_training_sesssions / 24)) * session) + 1)), 100)
            else:
                starting_charge[agent] += 1000 * random.randrange(-1, 1)
        elapsed_time = time.time() - start_time
        print(f"Get Starting Battery: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")


        start_time = time.time()

        all_routes = [None for route in coords]
        for index, (city_lat, city_long) in enumerate(coords):
            all_routes[index] = [get_org_dest_coords((city_lat, city_long), radius, seeds + i + index) for i in range(num_of_agents)]

        elapsed_time = time.time() - start_time
        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        envs = [None for route in all_routes]
        for index,  routes in enumerate(all_routes):
            envs[index] = EVSimEnvironment(max_num_timesteps, num_episodes, num_of_chargers, make, model, starting_charge, max_charge, routes, seeds + index)

        elapsed_time = time.time() - start_time
        print(f"Initialize Environment: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        user_input = ""

        while user_input != 'Done':
            if train_model:

                if user_input != "":
                    num_episodes = int(user_input)
                    start_from_previous_session = True
                    epsilon = 0.1

                state_dimension = num_of_chargers * 6 + 3   # Traffic level and distance of each station plus total charger num, total distance, and number of EVs
                action_dimension = num_of_chargers * 3      # attributes for each station

                print(f"Training using Deep-Q Learning - Session {session}")

                rewards = [] # Array of [(avg_reward, aggregation_num, route_index, seed)]
                output_values = [] # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
                global_weights = None

                for aggregate_step in range(aggregation_count):

                    local_weights_list = []
                    for ind, env in enumerate(envs):
                        local_weights, avg_rewards, avg_output_values = train_dqn(env, global_weights, aggregate_step, ind, seeds, thread_num, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes,
                              batch_size, buffer_limit, state_dimension, action_dimension, num_of_agents,
                              start_from_previous_session, layers, fixed_attributes)

                        rewards.append(avg_rewards)
                        output_values.append(avg_output_values)
                        local_weights_list.append(local_weights)

                    # Aggregate the weights from all local models
                    global_weights = aggregate_weights(local_weights_list)

                    print(f"\n\n############ Aggregation Step {aggregate_step} ############\n\n")

                if plot_aggregate_rewards:
                    plot_aggregate_reward_data(rewards)
                    plot_aggregate_output_values_per_route(output_values)

            if fixed_attributes != [0, 1] and fixed_attributes != [1, 0] and fixed_attributes != [0.5, 0.5]:
                attr_label = 'learned'
            else:
                attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

            if save_data:
                # Add this before you save your model
                if not os.path.exists('outputs'):
                    os.makedirs('outputs')

                for index, env in enumerate(envs):
                    env.write_path_to_csv(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    env.write_chargers_to_csv(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    env.write_reward_graph_to_csv(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    env.write_charger_traffic_to_csv(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')

            if generate_plots:
                for index, env in enumerate(envs):
                    route_data = read_csv_data(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    charger_data = read_csv_data(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    reward_data = read_csv_data(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')
                    traffic_data = read_csv_data(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}_{index}.csv')

                    route_datasets = []
                    if num_of_agents == 1:
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

            if num_episodes != 1 and continue_training:
                user_input = input("More Episodes? ")
            else:
                user_input = 'Done'

if __name__ == '__main__':
    t1 = threading.Thread(target=train_rl_vrp_csp, args=(0,))
    # t2 = threading.Thread(target=train_rl_vrp_csp, args=(1,))
    # t3 = threading.Thread(target=train_rl_vrp_csp, args=(2,))

    t1.start()
    # t2.start()
    # t3.start()

    t1.join()
    # t2.join()
    # t3.join()