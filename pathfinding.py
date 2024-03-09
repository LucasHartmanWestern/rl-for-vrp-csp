import copy
from geolocation.maps_free import get_distance_and_time
import time

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

def build_graph(env, agent_index):

    usage_per_min = env.ev_info() / 60
    start_soc = env.base_soc[agent_index]
    max_soc = env.max_soc
    max_dist_from_start = start_soc / usage_per_min
    max_dist_on_full_charge = max_soc / usage_per_min

    vertices = ['origin', 'destination']
    edges = {'origin': {}, 'destination': {}}

    # Distance in minutes from destination to origin
    org_to_dest_time = get_distance_and_time((env.dest_lat[agent_index], env.dest_long[agent_index]), (env.org_lat[agent_index], env.org_long[agent_index]))[1] / 60
    if org_to_dest_time < max_dist_from_start:
        edges['origin']['destination'] = org_to_dest_time
        edges['destination']['origin'] = org_to_dest_time

    # Loop through all chargers
    for i in range(len(env.charger_coords[agent_index])):
        vertices.append(i + 1) # Track charger ID
        edges[i + 1] = {} # Add station to edges

        charger = env.charger_coords[agent_index][i]

        # Distance in minutes from charger to origin
        time_to_charger = get_distance_and_time((charger[1], charger[2]), (env.org_lat[agent_index], env.org_long[agent_index]))[1] / 60

        # If you can make it to charger from origin, log it in the graph
        if time_to_charger < max_dist_from_start:
            edges['origin'][i + 1] = time_to_charger
            edges[i + 1]['origin'] = time_to_charger

        # Distance in minutes from destination to origin
        charger_to_dest_time = get_distance_and_time((charger[1], charger[2]), (env.dest_lat[agent_index], env.dest_long[agent_index]))[1] / 60
        if charger_to_dest_time < max_dist_on_full_charge:
            edges[i + 1]['destination'] = charger_to_dest_time
            edges['destination'][i + 1] = charger_to_dest_time

        # Populate graph of individual charger
        for j in range(len(env.charger_coords[agent_index])):
            if i != j: # Ignore self reference
                other_charger = env.charger_coords[agent_index][j]

                # Distance in minutes
                time_to_other_charger = get_distance_and_time((charger[1], charger[2]), (other_charger[1], other_charger[2]))[1] / 60

                # If you can make it from one charger to another on full charge, log it
                if time_to_other_charger < max_dist_on_full_charge:
                    edges[i + 1][j + 1] = time_to_other_charger

    return vertices, edges

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