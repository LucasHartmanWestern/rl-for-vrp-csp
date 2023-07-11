import copy
import heapq
import math

from geolocation.maps_free import get_distance_and_time

def baseline(environment, algorithm='dijkstra', index=0):
    environment.tracking_baseline = True
    environment.reset()

    agent_index = index

    make, model, battery_percentage, distance_to_dest, *charger_distances = environment.state[agent_index]
    usage_per_min = environment.ev_info() / 60

    # Build graph of possible paths from chargers to each other, the origin, and destination
    verts, edges = build_graph(environment, agent_index)

    if algorithm == 'dijkstra':
        # Use Dijkstra's algorithm to get shortest paths from origin
        dist, previous = dijkstra((verts, edges), 'origin')
    elif algorithm == 'A*':
        # Use Dijkstra's algorithm to get shortest paths from origin
        dist, previous = a_star((verts, edges), 'origin', 'destination')

    path = []

    # If user can make it to destination, go straight there
    if edges['origin'].get('destination') is not None:
        path.append(('destination', 0))

    # Build path based on going to chargers first
    else:
        prev = previous['destination']
        cur = 'destination'

        # Populate path to travel
        while prev != None:
            time_needed = edges[cur][prev] # Find time needed to get to next step
            target_soc = time_needed * usage_per_min # Find SoC needed to get to next step

            path.append((cur, target_soc)) # Update path

            # Update step
            cur = copy.copy(prev)
            prev = previous[prev]

    path.reverse() # Put destination step at the end

    # Travel path using simulator
    for i in range(len(path)):
        done = False

        if path[i][0] != 'destination':
            # Go to charger
            while environment.is_charging[index] is not True and done is not True:
                next_state, reward, done = environment.step(path[i][0])

            # Charge to needed amount
            while environment.cur_soc[index] < path[i + 1][1] and done is not True:
                next_state, reward, done = environment.step(path[i][0])

        else:
            # Go to destination
            done = False
            while done is not True:
                next_state, reward, done = environment.step(0)

def build_graph(env, agent_index):
    usage_per_min = env.ev_info() / 60
    start_soc = env.base_soc
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


def a_star(graph, start, end):
    graph = graph[1]

    # Define heuristic function
    def h(n):
        if n == end:
            return 0
        else:
            return graph[n][end]

    # Initialize heap with start node and cost
    heap = [(h(start), start)]
    # Initiate costs dictionary with infinite cost for all nodes except start node
    costs = {node: float('inf') for node in graph}
    costs[start] = 0
    # Track paths
    paths = {start: [start]}
    # Visited nodes
    visited = set()

    while heap:
        (fn, node) = heapq.heappop(heap)
        visited.add(node)

        if node == end:
            return paths[node]

        for neighbour, cost in graph[node].items():
            if neighbour not in visited:
                old_cost = costs[neighbour]
                new_cost = costs[node] + cost

                if new_cost < old_cost:
                    costs[neighbour] = new_cost
                    paths[neighbour] = paths[node] + [neighbour]
                    heapq.heappush(heap, (new_cost + h(neighbour), neighbour))

    return None

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