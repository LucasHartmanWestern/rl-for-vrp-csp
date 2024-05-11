import copy
import numpy as np

def build_graph(agent_index, ev_info, unique_chargers, org_lat, org_long, dest_lat, dest_long):

    starting_charge_list = ev_info['starting_charge'] # 5000-7000
    max_charge_list = ev_info['max_charge'] # in Watts
    usage_per_hour_list = ev_info['usage_per_hour'] # in Wh/100 km

    model_types = ev_info['model_type']
    model_indices = ev_info['model_indices']

    # Usage rates
    usage_per_min = usage_per_hour_list[agent_index] / 60
    start_soc = starting_charge_list[agent_index]
    max_soc = max_charge_list[agent_index]

    # Thresholds
    max_dist_from_start = start_soc / usage_per_min
    max_dist_on_full_charge = max_soc / usage_per_min

    # Convert unique_chargers to numpy array
    charger_locs = np.array([(lat, lon) for _, lat, lon in unique_chargers])
    all_points = np.vstack((charger_locs, [org_lat, org_long], [dest_lat, dest_long]))

    # Initialize graph matrix
    num_points = len(all_points)
    graph = np.zeros((num_points, num_points))

    # Populate graph matrix
    latitudes = np.radians(all_points[:, 0])
    longitudes = np.radians(all_points[:, 1])
    delta_lat = latitudes[:, np.newaxis] - latitudes
    delta_lon = longitudes[:, np.newaxis] - longitudes
    a = np.sin(delta_lat / 2) ** 2 + np.cos(latitudes)[:, np.newaxis] * np.cos(latitudes) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371  # Earth radius in kilometers
    graph = R * c

    # Threshold the edges so you cannot go below 0% battery
    graph[graph > max_dist_on_full_charge] = np.inf
    graph[len(unique_chargers), graph[len(unique_chargers)] > max_dist_from_start] = np.inf # Origin is capped based on the starting battery
    graph[:, len(unique_chargers)] = np.where(graph[:, len(unique_chargers)] > max_dist_from_start, np.inf, graph[:, len(unique_chargers)])

    return graph

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def dijkstra(graph):
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