import copy
import heapq
import math

from geolocation.maps_free import get_distance_and_time

def baseline(environment, algorithm='dijkstra', num_of_agents=0):
    print('Starting Baseline Calculations')

    environment.tracking_baseline = True
    environment.reset()

    paths = []

    for i in range(num_of_agents):
        make, model, battery_percentage, distance_to_dest, *charger_distances = environment.state[i]
        usage_per_min = environment.ev_info() / 60

        # Build graph of possible paths from chargers to each other, the origin, and destination
        verts, edges = build_graph(environment, i)

        if algorithm == 'dijkstra':
            # Use Dijkstra's algorithm to get shortest paths from origin
            dist, previous = dijkstra((verts, edges), 'origin')
        elif algorithm == 'A*':
            # Use A* algorithm to get shortest paths from origin
            dist, previous = a_star((verts, edges), 'origin', 'destination')
        elif algorithm == 'floyd-warshall':
            # Use Floyd-Warshall algorithm to get shortest paths from origin
            dist, previous = floyd_warshall((verts, edges))

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
                target_soc = time_needed * usage_per_min + usage_per_min # Find SoC needed to get to next step

                path.append((cur, target_soc)) # Update path

                # Update step
                cur = copy.copy(prev)
                prev = previous[prev]

        path.reverse() # Put destination step at the end

        paths.append(path)

    print('Baseline Paths Built')

    current_path = 0
    current_path_list = [i for i in range(len(paths))]

    while len(current_path_list) > 0:

        done = False

        # Check if step in path is completed
        if len(paths[current_path_list[current_path]]) > 1:
            if environment.is_charging[current_path_list[current_path]] is True and environment.cur_soc[current_path_list[current_path]] > paths[current_path_list[current_path]][1][1] + usage_per_min:
                print(f'DELETING STEP - {paths[current_path_list[current_path]][0]}')
                del paths[current_path_list[current_path]][0]

        if len(paths[current_path_list[current_path]]) > 0:
            print(f'STEP - {paths[current_path_list[current_path]][0]}')

            if paths[current_path_list[current_path]][0][0] != 'destination':
                next_state, reward, done = environment.step(paths[current_path_list[current_path]][0][0]) # Go to charger and charge until full
            else:
                next_state, reward, done = environment.step(0)

        if done is True:

            print(f'DELETING ROUTE - {current_path_list[current_path]} : REMAINING LENGTH - {len(current_path_list) - 1}')
            del current_path_list[current_path]

            if len(current_path_list) != 0:
                current_path = current_path % len(current_path_list)

        else:
            if len(current_path_list) > 0:
                current_path = (current_path + 1) % len(current_path_list)

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

def floyd_warshall(graph):
    vertices, edges = graph
    dist = dict()
    previous = dict()

    # Initialize distance and previous dictionaries
    for vertex in vertices:
        dist[vertex] = dict()
        previous[vertex] = dict()
        for other_vertex in vertices:
            if vertex == other_vertex:
                dist[vertex][other_vertex] = 0
                previous[vertex][other_vertex] = None
            elif other_vertex in edges[vertex]:
                dist[vertex][other_vertex] = edges[vertex][other_vertex]
                previous[vertex][other_vertex] = vertex
            else:
                dist[vertex][other_vertex] = float('inf')
                previous[vertex][other_vertex] = None

    # Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    previous[i][j] = previous[k][j]

    return dist, previous

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