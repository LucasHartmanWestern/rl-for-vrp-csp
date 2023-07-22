from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot, generate_traffic_plot
from geolocation.maps_free import get_org_dest_coords
from training_visualizer import Simulation
import random
import os
import time
import threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(thread_num):
    print(f'INITIALIZE THREAD {thread_num}')

    ############ Algorithm ############

    algorithm = "DQN"

    ############ Configuration ############

    train_model = True
    start_from_previous_session = False
    save_data = True
    generate_plots = True

    ############ Environment Settings ############

    seeds = [1000, 2000, 3000][thread_num] # Used for reproducibility
    num_of_agents = 100
    num_of_chargers = 3 # 3x this amount of chargers will be used (for origin, destination, and midpoint)
    make = 0 # Not currently used
    model = 0 # Not currently used
    max_charge = 100000 # 100kW
    city_lat, city_long = (42.983612, -81.249725) # Coordinates of city center
    radius = 20
    min_distance = 20
    starting_charge = [5000 for agent in range(num_of_agents)]

    ############ Hyperparameters ############

    num_training_sesssions = 1
    num_episodes = 5000
    epsilon = 0.9
    discount_factor = 0.999999
    epsilon_decay = (10 ** (-5 / (4 * num_episodes))) * ((1 / epsilon) ** (5 / (4 * num_episodes))) # Calculate decay such that by 4/5ths of the way through training, epsilon reaches 10%
    batch_size = max(round(num_episodes / 10), 1)
    max_num_timesteps = 500 # Amonut of minutes
    buffer_limit = max(num_episodes, 2) / 2 + batch_size
    layers = [128, 128, 64, 64, 32]

    ############ HPP Settings ############

    fixed_attributes = None # Assign fixed attributes to compare a baseline [Traffic_mult, Distance_mult]

    ############ Initialization ############

    # Run multiple training sessions with differing origins and destinations
    for session in range(num_training_sesssions):

        seeds += session

        if session != 0 and train_model:
            start_from_previous_session = True # Always continue from previous training session when running back-to-back sessions

        if radius is None:
            radius = ((1 / (num_training_sesssions / 14)) * session) + 1

        for agent in range(num_of_agents):
            random.seed(seeds + agent)
            # Random charge between 0.5-x%, where x scales between 1-25% as sessions continue
            if starting_charge is None:
                starting_charge[agent] = random.randrange(500, int(1000 * (((1 / (num_training_sesssions / 24)) * session) + 1)), 100)
            else:
                starting_charge[agent] += 1000 * random.randrange(-2, 2)

        # Get origin and destination coordinates, scale radius from center from 1-10km as sessions continue
        routes = [get_org_dest_coords((city_lat, city_long), radius, min_distance, seeds + i) for i in range(num_of_agents)]

        start_time = time.time()
        env = EVSimEnvironment(max_num_timesteps, num_episodes, num_of_chargers, make, model, starting_charge, max_charge, routes, seeds)
        elapsed_time = time.time() - start_time
        print(f"Initialize Environment: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        if train_model:

            state_dimension = num_of_chargers * 6 + 3   # Traffic level and distance of each station plus total charger num, total distance, and number of EVs
            action_dimension = num_of_chargers * 3      # attributes for each station

            if algorithm == "DQN":
                print(f"Training using Deep-Q Learning - Session {session}")
                train_dqn(env, seeds, thread_num, epsilon, epsilon_decay, discount_factor, num_episodes, batch_size, buffer_limit, state_dimension, action_dimension, num_of_agents, start_from_previous_session, layers, fixed_attributes)
            else:
                print(f"Training using Expected SARSA - Session {session}")
                train_sarsa(env, epsilon, discount_factor, num_episodes, epsilon_decay, max_num_timesteps, state_dimension, action_dimension - 1, num_of_agents, start_from_previous_session, seeds, layers)

        if fixed_attributes is None:
            attr_label = 'learned'
        else:
            attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

        if save_data:
            env.write_path_to_csv(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            env.write_chargers_to_csv(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            env.write_reward_graph_to_csv(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            env.write_charger_traffic_to_csv(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')

        if generate_plots:
            route_data = read_csv_data(f'outputs/routes_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            charger_data = read_csv_data(f'outputs/chargers_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            reward_data = read_csv_data(f'outputs/rewards_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')
            traffic_data = read_csv_data(f'outputs/traffic_{num_of_agents}_{num_episodes}_{seeds}_{attr_label}.csv')

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

            origins = [(route[0], route[1]) for route in routes]
            destinations = [(route[2], route[3]) for route in routes]
            generate_interactive_plot(algorithm, session, route_datasets, charger_data, origins, destinations)

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