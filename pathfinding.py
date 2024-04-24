import copy
from geolocation.maps_free import get_distance_and_time
import numpy as np

def build_path(environment, edges, dist, previous):

    path = []
    usage_per_min = environment.ev_info() / 60

    # If user can make it to destination, go straight there
    if edges['origin'].get('destination') is not None:
        path.append(('destination', 0))

    # Build path based on going to chargers first
    else:
        prev = previous['destination']
        cur = 'destination'

        # Populate path to travel
        while prev != None:
            time_needed = edges[cur][prev]  # Find time needed to get to next step

            target_soc = time_needed * usage_per_min + usage_per_min  # Find SoC needed to get to next step

            path.append((cur, target_soc))  # Update path

            # Update step
            cur = copy.deepcopy(prev)
            prev = previous[prev]

    path.reverse()  # Put destination step at the end

    return path

def build_graph(starting_charge, unique_chargers, org_lat, org_long, dest_lat, dest_long):

    # Usage rates
    usage_per_min = env.ev_info() / 60
    start_soc = env.base_soc[agent_index]
    max_soc = env.max_soc
    max_dist_from_start = start_soc / usage_per_min
    max_dist_on_full_charge = max_soc / usage_per_min

    # Initialize lists
    vertices = ['origin', 'destination']
    stations = []

    # Get origin/destination info
    origin_coords = (env.org_lat[agent_index], env.org_long[agent_index])  # Example coordinates
    destination_coords = (env.dest_lat[agent_index], env.dest_long[agent_index])  # Example coordinates
    stations.insert(0, ('origin', origin_coords[0], origin_coords[1]))
    stations.append(('destination', destination_coords[0], destination_coords[1]))

    # Loop through all chargers
    for i in range(len(env.charger_coords[agent_index])):
        vertices.append(i + 1) # Track charger ID
        charger = env.charger_coords[agent_index][i]
        stations.append(charger)

    # Number of locations including origin and destination
    n = len(stations)

    # Create the adjacency list from the distance matrix
    edges = {}
    for i in range(n):

        if i == 0:
            org_key = 'origin'
        elif i == 1:
            org_key = 'destination'
        else:
            org_key = (i - 1)

        edges[org_key] = {}

        for j in range(n):
            if i != j:
                if j == 0:
                    dest_key = 'origin'
                elif j == 1:
                    dest_key = 'destination'
                else:
                    dest_key = (j - 1)

                # Get distance
                dist = haversine(stations[i][1], stations[i][2], stations[j][1], stations[j][2])

                # Only add if possible to travel edge
                if i == 0 and dist <= max_dist_from_start:
                    edges[org_key][dest_key] = dist
                elif i > 0 and dist <= max_dist_on_full_charge:
                    edges[org_key][dest_key] = dist

    return vertices, edges

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def dijkstra(graph, source):

    vertices, edges = graph
    dist = dict()
    previous = dict()

    for vertex in vertices:
        dist[vertex] = float('inf')
        previous[vertex] = None

    dist[source] = 0
    vertices = set(vertices)

    while vertices:
        current_vertex = min(vertices, key=lambda vertex: dist[vertex])
        vertices.remove(current_vertex)

        if dist[current_vertex] == float('inf'):
            break

        for neighbour, cost in edges[current_vertex].items():
            alternative_route = dist[current_vertex] + cost
            if alternative_route < dist[neighbour]:
                dist[neighbour] = alternative_route
                previous[neighbour] = current_vertex

    return dist, previous