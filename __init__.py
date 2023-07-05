from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot
from baseline_algorithm import baseline
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

############ Algorithm ############

algorithm = "DQN"

############ Configuration ############

train_model = True
generate_baseline = False
start_from_previous_session = True
save_data = True
generate_plots = True

############ Environment Settings ############

num_of_chargers = 5 # 3x this amount of chargers will be used (for origin, destination, and midpoint)
make = 0 # Not currently used
model = 0 # Not currently used
starting_charge = 10000 # 50kW
max_charge = 100000 # 100kW
# 400 Lyle St, London
org_lat = 42.98881506714761
org_long = -81.22807778867828
# 7720 Patrick St, Port Franks
dest_lat = 43.23157243219816
dest_long = -81.88029292946138

############ Hyperparameters ############

num_episodes = 10000
epsilon = 0.50
discount_factor = 0.99999
batch_size = 128
max_num_timesteps = 120 # 2 hours
buffer_limit = (num_episodes * max_num_timesteps) / 2
layers = [32, 64, 128, 64, 32]

############ Initialization ############

env = EVSimEnvironment(num_episodes, num_of_chargers, make, model, starting_charge, max_charge, org_lat, org_long, dest_lat, dest_long)

if generate_baseline:
    baseline(env)

if train_model:

    state_dimension, action_dimension = env.get_state_action_dimension()

    if algorithm == "DQN":
        print("Training using Deep-Q Learning")
        train_dqn(env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers)
    else:
        print("Training using Expected SARSA")
        train_sarsa(env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers)

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