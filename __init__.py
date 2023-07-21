from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot, generate_traffic_plot
from geolocation.maps_free import get_org_dest_coords
from training_visualizer import Simulation
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

############ Algorithm ############

algorithm = "DQN"

############ Configuration ############

train_model = True
generate_baseline = False
start_from_previous_session = False
save_data = True
generate_plots = True
visualize_training = False

############ Environment Settings ############

seeds = 1 # Used for reproducibility
num_of_agents = 50
num_of_chargers = 3 # 3x this amount of chargers will be used (for origin, destination, and midpoint)
make = 0 # Not currently used
model = 0 # Not currently used
max_charge = 100000 # 100kW
city_lat, city_long = (42.983612, -81.249725) # Coordinates of city center
radius = 10
starting_charge = 1000 # 0.5%

############ Hyperparameters ############

num_training_sesssions = 1
num_episodes = 20000
epsilon = 0.8
discount_factor = 0.999999
epsilon_decay = (10 ** (-5 / (4 * num_episodes))) * ((1 / epsilon) ** (5 / (4 * num_episodes))) # Calculate decay such that by 4/5ths of the way through training, epsilon reaches 10%
batch_size = round(num_episodes / 10)
max_num_timesteps = 500 # Amonut of minutes
buffer_limit = num_episodes / 2 + batch_size
layers = [64, 64, 64, 32]

############ Initialization ############

# Run multiple training sessions with differing origins and destinations
for session in range(num_training_sesssions):

    seeds += session

    if session != 0 and train_model:
        start_from_previous_session = True # Always continue from previous training session when running back-to-back sessions

    if radius is None:
        radius = ((1 / (num_training_sesssions / 14)) * session) + 1

    # Random charge between 0.5-x%, where x scales between 1-25% as sessions continue
    if starting_charge is None:
        starting_charge = random.randrange(500, int(1000 * (((1 / (num_training_sesssions / 24)) * session) + 1)), 100)

    # Get origin and destination coordinates, scale radius from center from 1-10km as sessions continue
    routes = [get_org_dest_coords((city_lat, city_long), radius, seeds + i) for i in range(num_of_agents)]

    env = EVSimEnvironment(max_num_timesteps, num_episodes, num_of_chargers, make, model, starting_charge, max_charge, routes, seeds)

    if train_model:

        state_dimension = num_of_chargers * 6 + 3   # Traffic level and distance of each station plus total charger num, total distance, and number of EVs
        action_dimension = num_of_chargers * 3      # attributes for each station

        if algorithm == "DQN":
            print(f"Training using Deep-Q Learning - Session {session}")
            train_dqn(env, epsilon, epsilon_decay, discount_factor, num_episodes, batch_size, buffer_limit, state_dimension, action_dimension, num_of_agents, start_from_previous_session, layers)
        else:
            print(f"Training using Expected SARSA - Session {session}")
            train_sarsa(env, epsilon, discount_factor, num_episodes, epsilon_decay, max_num_timesteps, state_dimension, action_dimension - 1, num_of_agents, start_from_previous_session, seeds, layers)

    if save_data:
        env.write_path_to_csv('outputs/routes.csv')
        env.write_chargers_to_csv('outputs/chargers.csv')
        env.write_reward_graph_to_csv('outputs/rewards.csv')
        env.write_charger_traffic_to_csv('outputs/traffic.csv')

    if generate_plots:
        route_data = read_csv_data('outputs/routes.csv')
        charger_data = read_csv_data('outputs/chargers.csv')
        reward_data = read_csv_data('outputs/rewards.csv')
        traffic_data = read_csv_data('outputs/traffic.csv')

        route_datasets = []
        if num_of_agents == 1:
            for id_value, group in route_data.groupby('Episode Num'):
                route_datasets.append(group)
        else:
            for episode_num, episode_group in route_data.groupby('Episode Num'):
                if episode_num == route_data['Episode Num'].max():
                    for agent_num, agent_group in episode_group.groupby('Agent Num'):
                        route_datasets.append(agent_group)

        if train_model or start_from_previous_session:
            generate_average_reward_plot(algorithm, reward_data, session)

        if num_episodes == 1 and num_of_agents > 1:
            generate_traffic_plot(traffic_data)

        origins = [(route[0], route[1]) for route in routes]
        destinations = [(route[2], route[3]) for route in routes]
        generate_interactive_plot(algorithm, session, route_datasets, charger_data, origins, destinations)