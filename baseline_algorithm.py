import heapq
import math


def baseline(environment):
    environment.tracking_baseline = True
    environment.reset()

    make, model, battery_percentage, distance_to_dest, *charger_distances = environment.state

    distances = energy_efficient_path(environment, battery_percentage)
    print(distances)


def energy_efficient_path(environment, starting_battery_percentage):
    usage_per_hour, charge_per_hour = environment.ev_info()

    graph = create_graph(environment.charger_coords, ('origin', environment.org_lat, environment.org_long), ('destination', environment.dest_lat, environment.dest_long))

    start = 'origin'

    # Assume total battery charge can cover a distance equal to the full_charge_distance
    full_charge_distance = (environment.max_soc / (usage_per_hour)) * 60  # (kW / (kW/hr)) * km/h = km
    starting_distance = full_charge_distance * (starting_battery_percentage)

    print(f'Full Charge {full_charge_distance}\nStarting dist {starting_distance}\nBattery Percentage {starting_battery_percentage}')

    distances = {node: float('infinity') for node in graph}
    distances[start] = starting_distance

    pq = [(starting_distance, start)]

    while pq:
        curr_distance, curr_node = heapq.heappop(pq)

        if curr_distance > distances[curr_node]:
            continue

        for neighbor, weight in graph[curr_node].items():
            # weight here stands for the distance between curr_node and neighbor
            # it also represents the amount of energy to travel this distance
            distance_left = curr_distance - weight

            # If the remaining energy (distance_left) after reaching the neighbor is greater than
            # the current stored energy (distance) at the neighbor, update it
            if distance_left > distances[neighbor]:
                distances[neighbor] = distance_left
                heapq.heappush(pq, (distance_left, neighbor))

    return distances

def haversine(coord1, coord2):
    R = 6371 # Radius of the Earth in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def create_graph(array, origin, destination):
    graph = {}
    nodes = [origin] + array + [destination]
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            id1, lat1, lon1 = nodes[i]
            id2, lat2, lon2 = nodes[j]
            distance = haversine((lat1, lon1), (lat2, lon2))
            if id1 not in graph:
                graph[id1] = {}
            if id2 not in graph:
                graph[id2] = {}
            graph[id1][id2] = distance
            graph[id2][id1] = distance  # Assuming distance is the same in both directions
    return graph