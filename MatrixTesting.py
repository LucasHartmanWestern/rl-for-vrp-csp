import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
import cupy as cp
import warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


# Token (Nx2)
T = np.array([[43.02120034946083, -81.28349087468504],
              [43.004969336049854, -81.18631870502043],
              [42.95923445066671, -81.26016049362336],
              [42.98111190139387, -81.30953935839466],
              [42.9819404397449, -81.2508736429095],
              ])

# Battery percentage (Nx2)
B = np.array([0.20, 0.15, 0.11, 0.06, 0.21])

# Destinations (Mx2) (note M > N and the first N entries are the destinations of the Tokens)
D = np.array([[42.96477210152778, -81.20994279529941],
              [42.986248881938806, -81.17904374824452],
              [43.006964980419085, -81.33422562900907],
              [43.00432877393533, -81.16393754746214],
              [43.03294439058604, -81.26350114352788],
              [42.97520298007788, -81.3206637664334],
              [42.95950149646455, -81.2600673019313],
              [43.01217983061826, -81.27053864527043]
              ])

# Traffic level (Mx2)
Tr = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Actions for step k (NxM)
A = np.zeros((T.shape[0], D.shape[0]))

# Target battery level
Tar = np.array([[0.4, 0, 0],
                [0.5, 0, 0],
                [0.3, 0.4, 0],
                [0.2, 0, 0],
                [0.4, 0, 0]
                ])

# Stops NxP (P is the number of stops that are required for the route with the most stops)
S = np.array([[6, 1, 0],
              [7, 2, 0],
              [7, 8, 3],
              [7, 4, 0],
              [6, 5, 0]
              ])

# Step size (fixed distance)
step_size = 0.01

# Rate of battery increase
increase_rate = 0.1

# Rate of battery decrease
decrease_rate = 0.1

def update_actions(stops, target_battery_level, battery, destinations, token_locations):
    """
    Update the actions matrix based on the stops, target battery level, and current battery levels.
    """
    # Initialize the actions matrix with zeros
    actions = np.zeros((stops.shape[0], destinations.shape[0]))

    # Iterate through each token
    for i in range(stops.shape[0]):
        # Find the index of the next non-zero stop
        next_stop_index = np.where(stops[i] > 0)[0][0]
        next_stop = stops[i, next_stop_index]

        # Check if the token has reached its destination
        if np.allclose(token_locations[i], destinations[next_stop]):
            # Check if the battery level is sufficient
            if battery[i] >= target_battery_level[i, next_stop_index]:
                # Move to the next destination
                stops[i, next_stop_index] = 0  # Mark the current stop as visited
                actions[i, next_stop] = 0  # Stop moving to the current destination
                if next_stop_index + 1 < stops.shape[1]:
                    next_stop = stops[i, next_stop_index + 1]
                    actions[i, next_stop] = 1  # Start moving to the next destination
            else:
                # Continue charging at the current stop
                actions[i, next_stop] = 1
        else:
            # Continue moving towards the next stop
            actions[i, next_stop] = 1

    return actions

def move_tokens(actions, destinations, token_locations, step_size):
    """
    Move the tokens towards their destinations based on the actions matrix.
    """
    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            if actions[i, j] == 1:
                direction = destinations[j] - token_locations[i]
                distance = np.linalg.norm(direction)
                if distance > step_size:
                    token_locations[i] += direction / distance * step_size
                else:
                    token_locations[i] = destinations[j]
    return token_locations

def update_traffic(actions, traffic):
    """
    Update the traffic levels at the charging stations.
    """
    for i in range(actions.shape[1]):
        traffic[i] = np.sum(actions[:, i])
    return traffic

def update_battery(actions, destinations, token_locations, battery, charging_rate, discharge_rate):
    """
    Update the battery levels of the tokens.
    """
    for i in range(actions.shape[0]):
        if np.any(actions[i] == 1):
            battery[i] -= discharge_rate
        else:
            # Find the index of the charging station the token is currently at
            current_station = np.argmax(actions[i])
            if np.allclose(token_locations[i], destinations[current_station]):
                battery[i] += charging_rate
    return battery

def simulate(tokens, battery, destinations, traffic, target_battery_level, stops, step_size, k_steps):
    """
    Run the simulation for k steps.
    """
    token_coordinates = np.zeros((k_steps, tokens.shape[0], tokens.shape[1]))
    traffic_levels = np.zeros((k_steps, traffic.shape[0]))

    for k in range(k_steps):
        actions = update_actions(stops, target_battery_level, battery, destinations, tokens)
        tokens = move_tokens(actions, destinations, tokens, step_size)
        traffic = update_traffic(actions, traffic)
        battery = update_battery(actions, destinations, tokens, battery, charging_rate=0.01, discharge_rate=0.01)

        token_coordinates[k] = tokens
        traffic_levels[k] = traffic

    return token_coordinates, traffic_levels

def plot_token_paths(token_coords, destinations, stops):
    """
    Plot the paths of the tokens with charging stations, start points, and end points.
    """
    num_tokens = token_coords.shape[1]
    num_destinations = destinations.shape[0]

    # Plot the charging stations
    for i in range(num_destinations):
        plt.scatter(destinations[i, 0], destinations[i, 1], marker='s', s=100, label=f'Station {i + 1}')

    # Plot the paths of the tokens
    for i in range(num_tokens):
        plt.plot(token_coords[:, i, 0], token_coords[:, i, 1], marker='o', label=f'Token {i + 1}')
        # Mark the start point with a triangle
        plt.scatter(token_coords[0, i, 0], token_coords[0, i, 1], marker='^', s=100, color='black')
        # Mark the end point with a star
        plt.scatter(destinations[stops[i, 0], 0], destinations[stops[i, 0], 1], marker='*', s=100, color='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Token Paths and Charging Stations')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    token_cords, traffic_levels = simulate(T, B, D, Tr, Tar, S, step_size, 25)

    plot_token_paths(token_cords, D, S)

    print(traffic_levels)

    # start_time = time.time()
    # positions = move_tokens_gpu(T, D, A, step_size, A.shape[0])
    # end_time = time.time()
    #
    # duration = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    # print(f"The function took {duration:.10f} milliseconds to run.")
    #
    # visualize_movements(T, D, positions)