import numpy as np
import torch
DEBUG = False

def build_graph(agent_index, step_size, ev_info, unique_chargers, org_lat, org_long, dest_lat, dest_long, still_charging, device, dtype):
    """
    Builds a graph representing the distances between the origin, destination, and unique chargers for an agent.

    Parameters:
        agent_index (int): Index of the current agent.
        step_size (float): Amount to move each timestep
        ev_info (dict): Dictionary containing information about the electric vehicles, including starting charge,
                        maximum charge, and usage per hour.
        unique_chargers (list): List of tuples containing the unique charger IDs, latitudes, and longitudes.
        org_lat (float): Latitude of the origin.
        org_long (float): Longitude of the origin.
        dest_lat (float): Latitude of the destination.
        dest_long (float): Longitude of the destination.
        still_charging (int): 1 if the car was charging in the previous timestep and 0 otherwise

    Returns:
        numpy.ndarray: A graph represented as a 2D numpy array where each element represents the distance between points.
    """
    starting_charge_list = ev_info['starting_charge'] # 5000-7000
    max_charge_list = ev_info['max_charge'] # in Watts
    usage_per_hour_list = ev_info['usage_per_hour'] # in Wh/60 km

    model_types = ev_info['model_type']
    model_indices = ev_info['model_indices']

    # Usage rates
    usage_per_min = usage_per_hour_list[agent_index] / 60
    start_soc = starting_charge_list[agent_index]
    max_soc = max_charge_list[agent_index]

    # Thresholds
    max_steps_from_start = start_soc / usage_per_min
    max_steps_on_full_charge = max_soc / usage_per_min

    # Convert unique_chargers to numpy array
    charger_locs = torch.tensor([(lat, lon) for _, lat, lon in unique_chargers], device=device, dtype=dtype)
    all_points = torch.vstack((charger_locs, torch.tensor([org_lat, org_long], device=device, dtype=dtype), torch.tensor([dest_lat, dest_long], device=device, dtype=dtype)))

    if DEBUG:
        print(f"{agent_index} - ALL POINTS - {all_points}")

    # Initialize graph matrix
    num_points = len(all_points)
    graph = torch.zeros((num_points, num_points), device=device, dtype=dtype)

    # Populate graph matrix
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                # Calculate Euclidean distance between point i and point j
                distance = torch.linalg.norm(all_points[i] - all_points[j])
                # Calculate number of steps and store in graph
                graph[i, j] = ((distance / step_size) * usage_per_min) + usage_per_min

    # Apply thresholds
    graph[graph > max_soc] = torch.inf

    graph[len(unique_chargers), graph[len(unique_chargers)] > start_soc] = torch.inf  # Origin is capped based on the starting battery
    graph[:, len(unique_chargers)] = torch.where(graph[:, len(unique_chargers)] > start_soc, torch.inf, graph[:, len(unique_chargers)])

    return graph

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
        lat1 (float or torch.Tensor): Latitude of the first point.
        lon1 (float or torch.Tensor): Longitude of the first point.
        lat2 (float or torch.Tensor): Latitude of the second point.
        lon2 (float or torch.Tensor): Longitude of the second point.

    Returns:
        torch.Tensor: The distance between the two points in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    if not isinstance(lat1, torch.Tensor):
        lat1, lon1, lat2, lon2 = map(torch.tensor, [lat1, lon1, lat2, lon2])
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    distance = R * c
    
    return distance

def dijkstra(graph: torch.Tensor, agent_idx: int, device: torch.device, dtype: torch.dtype) -> list:
    """
    Implements Dijkstra's algorithm using PyTorch for GPU acceleration
    to find the shortest path in a graph from the origin to the destination.

    Assumes the graph represents distances, with float('inf') for non-existent edges.

    Parameters:
        graph (torch.Tensor): A 2D tensor representing the distances between nodes.
                              graph[i, j] is the distance from node i to node j.
                              Use float('inf') for non-edges.
                              Expected shape: (N+2, N+2).
        agent_idx (int): The index of the agent (currently unused in this implementation).
        device (torch.device): The device to run the algorithm on (e.g., 'cuda').
        dtype (torch.dtype): The data type for calculations (e.g., torch.float32).

    Returns:
        list: A list of node indices representing the shortest path from the origin
              to the destination, excluding the origin and destination themselves.
              Returns an empty list if the destination is unreachable.

    Notes:
        - The graph size includes two extra nodes: origin and destination.
        - Nodes 0 to N-1 are intermediate nodes.
        - Node N is the origin.
        - Node N+1 is the destination.
        - The input 'graph' tensor should already be on the target 'device'.
    """
    num_nodes = graph.shape[0]
    N = num_nodes - 2  # Number of intermediate nodes
    origin = N         # Index of the origin node
    destination = N + 1 # Index of the destination node

    # Distance from origin to all other nodes
    min_dist = torch.full((num_nodes,), float('inf'), device=device, dtype=dtype)
    min_dist[origin] = 0
    # Keep track of visited nodes
    visited = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    # To store the path
    path = torch.full((num_nodes,), -1, device=device, dtype=torch.long)


    # Main loop - iterate through nodes
    for _ in range(num_nodes):
        # --- Find the unvisited node 'u' with the smallest distance ---
        dist_masked = min_dist.clone()
        # Ignore visited nodes for minimum search
        dist_masked[visited] = float('inf')
        min_val, u = torch.min(dist_masked, dim=0)

        # Stop if remaining nodes are unreachable or all visited
        if min_val == float('inf'):
            break

        # Mark the current node as visited
        visited[u] = True

        # --- Update distances and path for neighbors ---
        # Calculate potential new distances through node 'u'
        distances_from_u = graph[u]
        new_dist = min_dist[u] + distances_from_u

        # Create mask for nodes to update: must be unvisited, reachable from u, and the new path must be shorter
        update_mask = (~visited) & (distances_from_u != float('inf')) & (new_dist < min_dist)

        # Apply updates using the mask
        min_dist[update_mask] = new_dist[update_mask]
        path[update_mask] = u


    # --- Path Reconstruction ---
    # Check if the destination node was reached
    if min_dist[destination] == float('inf'):
        print("Destination is unreachable.")
        return []

    # Transfer path to CPU and convert to numpy for easier processing
    path_cpu = path.cpu().numpy()
    dest_idx = destination
    rev_path = [] # Store the path in reverse order initially

    # Trace the path back from destination to origin
    while dest_idx != -1:
        rev_path.append(int(dest_idx)) # Add current node index
        if dest_idx == origin: # Stop if we reached the origin
             break
        # Move to the predecessor node
        prev_dest_idx = dest_idx
        dest_idx = path_cpu[dest_idx]
         # Basic check to prevent infinite loops in case of unexpected path data
        if dest_idx == prev_dest_idx and dest_idx != origin:
             print(f"Error during path reconstruction: stuck at node {dest_idx}") # Optional
             return []

    # Reverse the path to get the correct order from origin to destination
    shortest_path = rev_path[::-1]

    # Basic validation of the reconstructed path
    if not shortest_path or shortest_path[0] != origin:
        print("Error: Path reconstruction did not start at the origin.") # Optional
        return []
    if shortest_path[-1] != destination:
         pass

    # Return the path excluding the origin (first element) and destination (last element)
    return shortest_path[1:-1]