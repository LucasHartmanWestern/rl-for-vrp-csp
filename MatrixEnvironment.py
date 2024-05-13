import matplotlib.pyplot as plt
import matplotlib
import warnings
import torch
import numpy as np
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
else:
    print("CUDA is not available. Using CPU.")

# Token (Nx2)
T = torch.tensor([[43.02120034946083, -81.28349087468504],
              [43.004969336049854, -81.18631870502043],
              [42.95923445066671, -81.26016049362336],
              [42.98111190139387, -81.30953935839466],
              [42.9819404397449, -81.2508736429095],
              ])

# Battery percentage (Nx2)
B = torch.tensor([0.20, 0.15, 0.11, 0.26, 0.21])

# Destinations (Mx2) (note M > N and the first N entries are the destinations of the Tokens)
D = torch.tensor([[42.96477210152778, -81.20994279529941],
              [42.986248881938806, -81.17904374824452],
              [43.006964980419085, -81.33422562900907],
              [43.00432877393533, -81.16393754746214],
              [43.03294439058604, -81.26350114352788],
              [42.97520298007788, -81.3206637664334],
              [42.95950149646455, -81.2600673019313],
              [43.01217983061826, -81.27053864527043]
              ])

# Maximum amount of cars at each station before charging rate decreases
Cap = torch.tensor([5, 10, 3])

# Traffic level (Mx1)
Tr = torch.zeros(D.shape[0])

# Track which cars are going to move each timestep
Mov = torch.ones(T.shape[0])

# Actions for step k (NxM)
A = torch.zeros((T.shape[0], D.shape[0]))

# Target battery level
Tar = torch.tensor([[0, 0, 0],
                [0.5, 0, 0],
                [0.3, 0.5, 0],
                [0.2, 0, 0],
                [0.4, 0, 0]
                ])

# Stops NxP (P is the number of stops that are required for the route with the most stops)
S = torch.tensor([[1, 0, 0],
              [7, 2, 0],
              [7, 8, 3],
              [7, 4, 0],
              [6, 5, 0]
              ])

# Step size (fixed distance)
step_size = 0.001

# Rate of battery increase
ir = 0.01

# Rate of battery decrease
dr = 0.001

def get_actions(actions, stops_np):

    # Make new identity matrix
    max_val = len(actions[0])
    identity_matrix = torch.eye(max_val)
    stops = torch.from_numpy(stops_np)

    actions = (stops[:, 0].reshape(-1, 1) == torch.arange(max_val).float() @ identity_matrix).int()
    return actions

def get_distance(tokens, destinations, actions):

    # Compute targets for each token
    token_targets = actions @ destinations

    # Compute Euclidian distance for each token
    dist = ((tokens - token_targets).t() * torch.max(actions, axis=1).values).t()
    total_dist = torch.sqrt(torch.sum(dist ** 2, axis=1)).reshape(-1, 1)

    # Return distances as Nx1 matrix
    return total_dist.reshape(-1)

def get_arrived(dists, dist_threshold):

    # Return 1 if below threshold, and 0 if above
    return (dists <= dist_threshold).int()

def get_traffic(stops_np, destinations, is_charging):

    # Get stations to charge at
    target_stations = torch.from_numpy(stops_np[:, 0]).reshape(-1) + 1

    # Isolate stations to charge at
    stations_in_use = (target_stations *is_charging).int()
    stations_in_use = torch.maximum(stations_in_use - 1, 0)

    # Used to find end of the list
    max_num = destinations.shape[0]

    # I don't really know how this works, but it does
    traffic_level = torch.bincount(stations_in_use, minlength=max_num)

    # Ignore destination 0
    traffic_level[0] = 0

    # Note that the first N entries can be ignored since they aren't charging stations, but rather final destinations
    return traffic_level

def get_charging_rates(stops, traffic_level, arrived, capacity, decrease_rate, increase_rate):

    # Get target stop for each car
    target_stop = stops[:, 0]

    # Get charging rate for each station
    station_rate = torch.minimum((capacity / torch.maximum(traffic_level, 1)) * increase_rate, increase_rate)

    # Offset everything by decrease rate
    station_rate += decrease_rate

    # Get charging rates for each car based on target
    rates_by_car = station_rate[target_stop.astype(int)]

    # Zero out charging rate for cars that haven't arrived
    rates_by_car *= arrived.astype(int)

    # Undo the offset
    rates_by_car -= decrease_rate

    # Zero-out charging rate for cars already at their destination
    diag_matrix = torch.diag([0 if x == -1 else 1 for x in target_stop])
    rates_by_car = rates_by_car @ diag_matrix

    # Return Nx1 charging rate for each car
    return rates_by_car

def update_battery(battery, charge_rate):
    return battery + charge_rate

def get_battery_charged(battery, target):

    # Target for this stop
    stop_target = target[:, 0]

    # Return if the battery level is above the needed level to depart from charging station
    return (battery >= stop_target).astype(int)

def move_tokens(tokens_np, moving, actions, destinations, step_size):

    # Get target destinations for tokens that are moving
    target_coords = actions @ destinations
    tokens = torch.from_numpy(tokens_np)

    # Compute the displacement vectors and their norms
    right_term = (target_coords - tokens).t()
    left_term = torch.max(actions, axis=1).values
    displacement = (right_term * left_term).t()
    distances = torch.linalg.norm(displacement, axis=1)

    # Compute the normalized direction vectors
    # Initialize the output tensor with zeros, matching the shape of displacement
    direction = torch.zeros_like(displacement)
    # Create a mask where distances are not zero
    mask = distances != 0
    # Perform the division where the mask is True
    direction[mask] = displacement[mask] / distances[mask].unsqueeze(1)

    # Zero-out direction for tokens that aren't moving
    masked_direction = (direction.t() * moving).t()

    # Compute the step vectors
    steps = torch.minimum(distances, torch.tensor(step_size, dtype=distances.dtype)).unsqueeze(-1)
    steps = steps * masked_direction
    
    # If the car is charging but isn't directly on the charging station, shift them onto it
    shift_to_charger_while_charging = (displacement.t() * ((moving - 1) * -1)).t()

    # Update token positions
    new_tokens = tokens + steps
    new_tokens += shift_to_charger_while_charging

    # Calculate distance travelled
    distance_travelled = torch.sqrt(torch.sum((tokens - new_tokens) ** 2, axis=1))

    return new_tokens, distance_travelled

def update_stops(stops, ready_to_leave):

    # Create transformation matrix
    K = stops.shape[1]
    transform_matrix = torch.zeros((K, K), dtype=torch.float32)
    transform_matrix[torch.arange(1, K), torch.arange(K - 1)] = 1

    ready_to_leave = ready_to_leave.reshape(-1, 1)

    # I'm not sure how this works tbh...
    updated_stops =  stops * ready_to_leave @ transform_matrix + stops * (1 - ready_to_leave)

    # Set rows to zeros if there is only one nonzero element in the row and ready_to_leave is 1
    for i in range(len(stops)):
        if ready_to_leave[i] == 1 and torch.count_nonzero(stops[i]) == 1:
            updated_stops[i] = torch.zeros(K)

    return updated_stops

def simulate_matrix_env(tokens, battery, destinations, actions, moving, traffic_level, capacity_np, target_battery_level, stops, step_size, increase_rate, decrease_rate, k_steps):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capacity = torch.from_numpy(capacity_np)
    
    # Pre-process capacity array
    capacity = torch.concatenate((torch.zeros(tokens.shape[0]), capacity))

    step_count = 0

    paths = []
    traffic_per_charger = []
    battery_levels = []
    distances_per_car = [torch.zeros(tokens.shape[0])]

    while max(stops[:,0]) > 0 and step_count <= k_steps:

        stops[:,0] -= 1

        # Get NxM matrix of actions
        actions = get_actions(actions, stops)

        # Move the tokens and get the update position
        tokens, distance_travelled = move_tokens(tokens, moving, actions, destinations, step_size)

        # Track token position at each timestep and how far they travelled
        paths.append(tokens)
        distances_per_car.append(distance_travelled + distances_per_car[-1])

        # Get Nx1 matrix of distances
        distances = get_distance(tokens, destinations, actions)

        # Get Nx1 matrix of 0s or 1s that indicate if a car has arrived at current stop
        arrived = get_arrived(distances, step_size)

        # Accumulate traffic level of each station as Mx1 matrix
        traffic_level = get_traffic(stops, destinations, arrived)

        # Track traffic for each timestep
        traffic_per_charger.append(traffic_level)

        # Get charging or discharging rate for each car as Nx1 matrix
        charging_rates = get_charging_rates(stops, traffic_level, arrived, capacity, decrease_rate, increase_rate)

        # Update the battery level of each car
        battery = update_battery(battery, charging_rates)

        battery_levels.append(battery)

        # Check if the car is at their target battery level
        battery_charged = get_battery_charged(battery, target_battery_level)

        # Charging but ready to leave
        ready_to_leave = battery_charged * arrived

        # Charging and not ready to leave
        not_ready_to_leave = arrived - ready_to_leave

        # Update which cars will move
        moving = (not_ready_to_leave - 1) * -1

        # Zero-out tokens that are already at their stop
        diag_matrix = torch.diag([0 if x == -1 else 1 for x in stops[:, 0]])
        moving = moving @ diag_matrix

        stops[:, 0] += 1

        # Change the stops array to shift over the next stop if the token is ready to leave
        stops = update_stops(stops, ready_to_leave)
        target_battery_level = update_stops(target_battery_level, ready_to_leave)

        # Increase step count
        step_count += 1

    return paths, traffic_per_charger, battery_levels, distances_per_car


def visualize_simulation(paths, destinations):
    # Rearrange the data so that each path represents the movement of one token over time
    token_paths = [torch.array([step[i] for step in paths]) for i in range(len(paths[0]))]

    # Plot the token paths
    colors = ['blue', 'green', 'orange', 'cyan', 'magenta']
    for i, path in enumerate(token_paths):
        plt.plot(path[:, 1], path[:, 0], marker='o', markersize=3, linestyle='-', color=colors[i], label=f'Token {i + 1}')

    # Plot the destinations
    plt.scatter(destinations[:, 1], destinations[:, 0], marker='*', s=100, c='red', label='Destinations')

    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Token Paths and Destinations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def visualize_stats(data, title, measure):
    num_steps = len(data)
    num_values = len(data[0])

    # Create a time axis
    time = torch.arange(num_steps)

    # Plot the change in values for each index
    for i in range(num_values):
        values = [array[i] for array in data]
        plt.plot(time, values, label=f'{measure} {i + 1}')

    # Set labels and title
    plt.xlabel('Step')
    plt.ylabel(f'{measure}')
    plt.title(f'{title}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':

    # Run simulation
    paths, traffic, battery_levels, distances = simulate_matrix_env(T, B, D, A, Mov, Tr, Cap, Tar, S, step_size, ir, dr, 500)

    # Show the paths of each car
    visualize_simulation(paths, D)

    # Plot the various stats
    visualize_stats(traffic, 'Change in Traffic Levels Over Time', 'Traffic Level')
    visualize_stats(battery_levels, 'Change in Battery Level Over Time', 'Battery Level')
    visualize_stats(distances, 'Distance Travelled Over Time', 'Distance Travelled')