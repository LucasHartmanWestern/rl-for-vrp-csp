# Created by Lucas
# Restructure by Santiago 03/07/2024

#Helper functions to route and update to simulate routing in the environment class
import torch
import numpy as np

def get_actions(actions, stops, dtype):

    """
    Generates a new identity matrix and creates action vectors based on the stops.

    Parameters:
        actions (torch.Tensor): A tensor representing possible actions.
        stops (torch.Tensor): A tensor containing the stops for each action.

    Returns:
        torch.Tensor: A tensor containing the action vectors.
    """

    # Make new identity matrix
    max_val = actions.shape[1]
    identity_matrix = torch.eye(max_val, dtype=dtype, device=stops.device)

    left_matrix = (stops[:, 0].reshape(-1, 1) == torch.arange(max_val, dtype=dtype, device=stops.device))
    actions = torch.matmul(left_matrix.to(dtype), identity_matrix).int()
    
    #removing unused tensors from memory
    identity_matrix = None
    left_matrix = None
    
    #This function needs confirmation of use between int and float types
    return actions.to(dtype)

def get_distance(tokens, destinations, actions):

    """
    Computes the Euclidean distance between each token's position and its corresponding target destination.

    Parameters:
        tokens (torch.Tensor): A tensor containing the coordinates of the tokens.
        destinations (torch.Tensor): A tensor containing the coordinates of the possible destinations.
        actions (torch.Tensor): A tensor indicating the chosen actions for each token.

    Returns:
        torch.Tensor: A tensor containing the distances between each token and its target destination as a 1D array.
    """

    # Compute targets for each token
    token_targets = torch.matmul(actions, destinations)

    # Compute Euclidian distance for each token
    dist = ((tokens - token_targets) * torch.max(actions, axis=1).values.unsqueeze(1))
    total_dist = torch.sqrt(torch.sum(dist ** 2, axis=1)).reshape(-1, 1)

    # Return distances as Nx1 matrix
    return total_dist.reshape(-1)

def get_arrived(dists, dist_threshold):

    """
    Determines whether each distance is below or above a given threshold.

    Parameters:
        dists (torch.Tensor): A tensor containing distances.
        dist_threshold (float): The distance threshold.

    Returns:
        torch.Tensor: A tensor containing 1s and 0s where 1 indicates the distance is below or equal to the threshold
                      and 0 indicates the distance is above the threshold.
    """

    # Return 1 if below threshold, and 0 if above
    return (dists <= dist_threshold).int()

def get_traffic(stops, destinations, is_charging):

    """
    Calculates the traffic level at each charging station based on the stops and charging status of vehicles.

    Parameters:
        stops (torch.Tensor): A tensor containing the stops for each vehicle.
        destinations (torch.Tensor): A tensor containing the coordinates of possible destinations, including charging stations.
        is_charging (torch.Tensor): A tensor indicating whether each vehicle is charging (1) or not (0).

    Returns:
        torch.Tensor: A tensor containing the traffic level at each charging station.
    """

    # Get stations to charge at
    target_stations = stops[:, 0].reshape(-1) + 1

    # Isolate stations to charge at
    stations_in_use = (target_stations *is_charging).int()
    stations_in_use = torch.maximum(stations_in_use - 1, torch.tensor(0, dtype=stations_in_use.dtype))


    # Used to find end of the list
    max_num = destinations.shape[0]

    # I don't really know how this works, but it does
    traffic_level = torch.bincount(stations_in_use, minlength=max_num)

    # Ignore destination 0
    traffic_level[0] = 0

    # Note that the first N entries can be ignored since they aren't charging stations, but rather final destinations
    return traffic_level

def get_charging_rates(stops, traffic_level, arrived, capacity, decrease_rates, increase_rate, dtype):
    """
    Calculates the charging rates for each vehicle based on their current stops, traffic level at charging stations,
    arrival status, and station capacities.

    Parameters:
        stops (torch.Tensor): A tensor containing the stops for each vehicle.
        traffic_level (torch.Tensor): A tensor containing the traffic level at each charging station.
        arrived (torch.Tensor): A tensor indicating whether each vehicle has arrived at its target stop (1 if arrived, 0 otherwise).
        capacity (torch.Tensor): A tensor containing the capacity of each charging station.
        decrease_rates (torch.Tensor): A tensor containing the rate at which charging decreases by model.
        increase_rate (float): The maximum rate at which charging can increase.

    Returns:
        torch.Tensor: A tensor containing the charging rates for each vehicle.
    """

    # Get target stop for each car
    target_stop = stops[:, 0]

    # Get charging rate for each station
    max_traffic_level = torch.maximum(traffic_level, torch.tensor(1.0, dtype=traffic_level.dtype))
    capacity_rate = (capacity / max_traffic_level) * increase_rate
    station_rate = torch.minimum(capacity_rate, torch.tensor(increase_rate, dtype=capacity_rate.dtype, device=capacity_rate.device))

    # Offset everything by decrease rate
    station_rate = station_rate.unsqueeze(1) + decrease_rates.unsqueeze(0)

    # Get charging rates for each car based on target
    rates_by_car = station_rate[target_stop.long(), torch.arange(len(decrease_rates)).long()]

    # Zero out charging rate for cars that haven't arrived
    rates_by_car *= arrived.int()

    # Undo the offset
    rates_by_car -= decrease_rates

    # Zero-out charging rate for cars already at their destination
    diag_matrix = torch.diag(torch.tensor([0 if x == -1 else 1 for x in target_stop], dtype=dtype, device=capacity.device))
    rates_by_car = torch.matmul(rates_by_car.to(dtype), diag_matrix.to(dtype))

    #Cleaning unused tensors
    capacity_rate = None
    station_rate = None
    diag_matrix = None
    
    
    # Return Nx1 charging rate for each car
    return rates_by_car

def update_battery(battery, charge_rate):

    """
    Updates the battery levels of vehicles based on the charging rates.

    Parameters:
        battery (torch.Tensor): A tensor containing the current battery levels of the vehicles.
        charge_rate (torch.Tensor): A tensor containing the charging rates for the vehicles.

    Returns:
        torch.Tensor: A tensor containing the updated battery levels.
    """

    return battery + charge_rate

def get_battery_charged(battery, target):

    """
    Determines whether each vehicle's battery level is sufficient to depart from the charging station.

    Parameters:
        battery (torch.Tensor): A tensor containing the current battery levels of the vehicles.
        target (torch.Tensor): A tensor containing the target battery levels required to depart from the charging station.

    Returns:
        torch.Tensor: A tensor containing 1s and 0s where 1 indicates the battery level is sufficient and 0 indicates it is not.
    """
    # Return if the battery level is above the needed level to depart from charging station
    return (battery >= target[:, 0]).int()

def move_tokens(tokens, moving, actions, destinations, step_size):

    """
    Moves tokens towards their target destinations, updating their positions and calculating the distance traveled.

    Parameters:
        tokens (torch.Tensor): A tensor containing the current positions of the tokens.
        moving (torch.Tensor): A tensor indicating whether each token is moving (1 if moving, 0 otherwise).
        actions (torch.Tensor): A tensor indicating the chosen actions for each token.
        destinations (torch.Tensor): A tensor containing the coordinates of the possible destinations.
        step_size (float): The maximum step size for each movement.

    Returns:
        tuple: A tuple containing:
            - new_tokens (torch.Tensor): A tensor containing the updated positions of the tokens.
            - distance_travelled (torch.Tensor): A tensor containing the distances traveled by each token.
    """

    # Get target destinations for tokens that are moving
    target_coords = torch.matmul(actions, destinations)

    # Compute the displacement vectors and their norms
    displacement = torch.mul((target_coords - tokens), torch.max(actions, axis=1).values.unsqueeze(1))
    distances_to_travel = torch.linalg.norm(displacement, axis=1)

    # Compute the normalized direction vectors
    # Initialize the output tensor with zeros, matching the shape of displacement
    direction = torch.zeros_like(displacement)
    # Create a mask where distances are not zero
    mask = distances_to_travel != 0
    # Perform the division where the mask is True
    direction[mask] = displacement[mask] / distances_to_travel[mask].unsqueeze(1)

    # Zero-out direction for tokens that aren't moving
    masked_direction = torch.mul(direction, moving.unsqueeze(-1))
    
    # Compute the step vectors
    steps = torch.minimum(distances_to_travel, torch.tensor(step_size, dtype=distances_to_travel.dtype)).unsqueeze(1)
    steps = steps * masked_direction
    
    # If the car is charging but isn't directly on the charging station, shift them onto it
    shift_to_charger_while_charging = torch.mul(displacement, ((moving - 1) * -1).unsqueeze(1))

    # Update token positions
    new_tokens = tokens + steps
    new_tokens += shift_to_charger_while_charging

    # Calculate distance travelled
    distance_travelled = torch.sqrt(torch.sum((tokens - new_tokens) ** 2, axis=1))

    return new_tokens, distance_travelled

def update_stops(stops, ready_to_leave, dtype):

    """
    Updates the stops for each vehicle based on their readiness to leave the current stop.

    Parameters:
        stops (torch.Tensor): A tensor containing the current stops for each vehicle.
        ready_to_leave (torch.Tensor): A tensor indicating whether each vehicle is ready to leave its current stop (1 if ready, 0 otherwise).

    Returns:
        torch.Tensor: A tensor containing the updated stops for each vehicle.
    """

    # Create transformation matrix
    K = stops.shape[1]
    transform_matrix = torch.zeros((K, K), dtype=dtype, device=stops.device)
    transform_matrix[torch.arange(1, K), torch.arange(K - 1)] = 1

    ready_to_leave = ready_to_leave.reshape(-1, 1)

    # I'm not sure how this works tbh...
    updated_stops = torch.matmul((stops * ready_to_leave.to(dtype)), transform_matrix.to(dtype)) + stops * (1 - ready_to_leave)
    
    # Set rows to zeros if there is only one nonzero element in the row and ready_to_leave is 1
    for i in range(len(stops)):
        if ready_to_leave[i] == 1 and torch.count_nonzero(stops[i]) == 1:
            updated_stops[i] = torch.zeros(K)
   
    return updated_stops