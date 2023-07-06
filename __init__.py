from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot
from baseline_algorithm import baseline
from geolocation.maps_free import get_org_dest_coords
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

############ Algorithm ############

algorithm = "SARSA"

############ Configuration ############

train_model = True
generate_baseline = False
start_from_previous_session = False
save_data = True
generate_plots = True

############ Environment Settings ############

seeds = 123 # Used for reproducibility
num_of_chargers = 5 # 3x this amount of chargers will be used (for origin, destination, and midpoint)
make = 0 # Not currently used
model = 0 # Not currently used
starting_charge = random.randrange(1000, 50000) # Random charge between 1-50%
max_charge = 100000 # 100kW
city_lat, city_long = (42.983612, -81.249725) # Coordinates of city center
org_lat, org_long, dest_lat, dest_long = get_org_dest_coords((city_lat, city_long)) # Get origin and destination coordinates

############ Hyperparameters ############

num_episodes = 5000
epsilon = 0.6
discount_factor = 0.9999
epsilon_decay = (10 ** (-5 / (4 * num_episodes))) * ((1 / epsilon) ** (5 / (4 * num_episodes))) # Calculate decay such that by 4/5ths of the way through training, epsilon reaches 10%
batch_size = 1000
max_num_timesteps = 50 # Amonut of minutes
buffer_limit = (num_episodes * max_num_timesteps) / 3 + batch_size
layers = [32, 64, 1024, 64, 32]

############ Initialization ############

env = EVSimEnvironment(max_num_timesteps, num_episodes, num_of_chargers, make, model, starting_charge, max_charge, org_lat, org_long, dest_lat, dest_long, seeds)

if generate_baseline:
    baseline(env)

if train_model:

    state_dimension, action_dimension = env.get_state_action_dimension()

    if algorithm == "DQN":
        print("Training using Deep-Q Learning")
        train_dqn(env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers)
    else:
        print("Training using Expected SARSA")
        train_sarsa(env, epsilon, discount_factor, num_episodes, epsilon_decay, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, seeds, layers)

if save_data:
    env.write_path_to_csv('outputs/routes.csv')
    env.write_chargers_to_csv('outputs/chargers.csv')
    env.write_reward_graph_to_csv('outputs/rewards.csv')

if generate_plots:
    route_data = read_csv_data('outputs/routes.csv')
    charger_data = read_csv_data('outputs/chargers.csv')
    reward_data = read_csv_data('outputs/rewards.csv')

    route_datasets = []
    for id_value, group in route_data.groupby('Episode Num'):
        route_datasets.append(group)

    if train_model or start_from_previous_session:
        generate_average_reward_plot(algorithm, reward_data)

    generate_interactive_plot(algorithm, route_datasets, charger_data, (org_lat, org_long), (dest_lat, dest_long))