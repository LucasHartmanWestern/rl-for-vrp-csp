import copy
import numpy as np

def build_graph(agent_index, step_size, ev_info, unique_chargers, org_lat, org_long, dest_lat, dest_long):
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
    print(f'agent {usage_per_hour_list}')
    start_soc = starting_charge_list[agent_index]
    max_soc = max_charge_list[agent_index]

    # Thresholds
    max_steps_from_start = start_soc / usage_per_min
    max_steps_on_full_charge = max_soc / usage_per_min

    # Convert unique_chargers to numpy array
    charger_locs = np.array([(lat, lon) for _, lat, lon in unique_chargers])
    all_points = np.vstack((charger_locs, [org_lat, org_long], [dest_lat, dest_long]))


    # Initialize graph matrix
    num_points = len(all_points)
    graph = np.zeros((num_points, num_points))

    # Populate graph matrix
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                # Calculate Euclidean distance between point i and point j
                distance = np.linalg.norm(all_points[i] - all_points[j])
                #print(f'distance: {distance}')
                #print(f'step_size: {step_size}')
                #print(f'usage per min: {usage_per_min}')
                # Calculate number of steps and store in graph
                graph[i, j] = (distance / step_size) * usage_per_min

    # Apply thresholds
    graph[graph > max_soc] = np.inf
    graph[len(unique_chargers), graph[len(unique_chargers)] > start_soc] = np.inf  # Origin is capped based on the starting battery
    graph[:, len(unique_chargers)] = np.where(graph[:, len(unique_chargers)] > start_soc, np.inf, graph[:, len(unique_chargers)])

    return graph

def haversine(lat1, lon1, lat2, lon2):

    """
    Calculates the great-circle distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The distance between the two points in kilometers.
    """

    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def dijkstra(graph, agent_index):

    """
    Implements Dijkstra's algorithm to find the shortest path in a graph from the origin to the destination.

    Parameters:
        graph (numpy.ndarray): A 2D array representing the distances between nodes in the graph.

    Returns:
        list: A list of node indices representing the shortest path from the origin to the destination, excluding the origin and destination themselves.
    """

    N = len(graph) - 2  # Exclude the origin and destination which are included in the graph size (N+2)
    origin = N  # Second last entry
    destination = N + 1  # Last entry

    # Distance from origin to all other nodes
    min_dist = [float('inf')] * len(graph)
    min_dist[origin] = 0
    visited = [False] * len(graph)
    path = [-1] * len(graph)  # To store the path

    def get_next_vertex():
        min_vertex = -1
        min_value = float('inf')
        for v in range(len(graph)):
            if not visited[v] and min_dist[v] < min_value:
                min_value = min_dist[v]
                min_vertex = v
        return min_vertex

    for _ in range(len(graph)):
        u = get_next_vertex()
        visited[u] = True

        for v in range(len(graph)):
            if graph[u][v] >= 0 and not visited[v]:
                if min_dist[u] + graph[u][v] < min_dist[v]:
                    min_dist[v] = min_dist[u] + graph[u][v]
                    path[v] = u

    # Extract the path to the destination
    def extract_path(dest):
        rev_path = []
        while dest != -1:
            rev_path.append(dest)
            dest = path[dest]
        return rev_path[::-1]

    return extract_path(destination)[1:-1]