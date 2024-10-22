# Created by Lucas
# Restructured by Santiago 03/07/2024

import torch
import numpy as np
import copy
import requests
import datetime
from geopy.distance import geodesic
import os
import csv

try:
    from ._helpers_routing import *
    from ._pathfinding import dijkstra, build_graph, haversine
    from .agent_info import agent_info
    from data_loader import load_config_file
except ImportError:
    print("Cannot import local files")


DEBUG = False

# Predefined list of supported cities with their coordinates and WeatherStats URLs
supported_cities = [
    {"name": "Charlottetown", "lat": 46.2382, "lon": -63.1311, "url": "https://charlottetown.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Edmonton", "lat": 53.5461, "lon": -113.4938, "url": "https://edmonton.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Fredericton", "lat": 45.9636, "lon": -66.6431, "url": "https://fredericton.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Halifax (Shearwater)", "lat": 44.6488, "lon": -63.5752, "url": "https://halifax.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Iqaluit", "lat": 63.7467, "lon": -68.5170, "url": "https://iqaluit.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Ottawa (Kanata - Orléans)", "lat": 45.4215, "lon": -75.6972, "url": "https://ottawa.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Québec", "lat": 46.8139, "lon": -71.2082, "url": "https://quebec.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Regina", "lat": 50.4452, "lon": -104.6189, "url": "https://regina.weatherstats.ca/data/temperature-daily.json"},
    {"name": "St. John's", "lat": 47.5615, "lon": -52.7126, "url": "https://stjohns.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Toronto", "lat": 43.65107, "lon": -79.347015, "url": "https://toronto.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Victoria", "lat": 48.4284, "lon": -123.3656, "url": "https://victoria.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Whitehorse", "lat": 60.7212, "lon": -135.0568, "url": "https://whitehorse.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Winnipeg", "lat": 49.8951, "lon": -97.1384, "url": "https://winnipeg.weatherstats.ca/data/temperature-daily.json"},
    {"name": "Yellowknife", "lat": 62.4540, "lon": -114.3718, "url": "https://yellowknife.weatherstats.ca/data/temperature-daily.json"}
]

def get_closest_city(lat: float, lon: float) -> dict:
    """
    Get the closest supported city for the given coordinates.

    Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        dict: Information about the closest supported city.
    """
    closest_city = None
    min_distance = float('inf')

    for city in supported_cities:
        city_lat = city['lat']
        city_lon = city['lon']
        distance = geodesic((lat, lon), (city_lat, city_lon)).kilometers

        if distance < min_distance:
            min_distance = distance
            closest_city = city

    if closest_city is None:
        raise Exception("No supported city found")

    return closest_city

def save_temps(coords_list: list, seed_list: list):
    # Ensure the directory exists
    os.makedirs('merl_env/temps', exist_ok=True)
    
    # Define the CSV file path
    csv_file_path = 'merl_env/temps/temperatures.csv'
    
    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header row
        csvwriter.writerow(['latitude', 'longitude', 'season', 'seed', 'temperature'])
        
        # Loop through every combination of coord, season, and seed then save the temperature to the CSV file
        for coords in coords_list:
            for seed in seed_list:
                for season in ['spring', 'summer', 'autumn', 'winter']:

                    temp = get_temperature(season, coords, np.random.default_rng(seed), seed)
                    csvwriter.writerow([coords[0], coords[1], season, seed, temp])

def get_temps_from_file(coords: list, seed: int, season: str):
    # Define the CSV file path
    csv_file_path = 'merl_env/temps/temperatures.csv'
    
    # Open the CSV file for reading
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Loop through the rows to find the matching entry
        for row in csvreader:
            if (float(row['latitude']) == coords[0] and 
                float(row['longitude']) == coords[1] and 
                row['season'] == season and 
                int(row['seed']) == seed):
                return float(row['temperature'])
    
    # If no matching entry is found, raise an exception
    raise Exception("Temperature data not found in CSV file")

def get_temperature(season: str, coords: list, rng: np.random.Generator, seed: int) -> float:
    """
    Get the average temperature for the given season and coordinates.

    Parameters:
        season (str): Season to get the temperature for.
        coords (list): Coordinates [latitude, longitude] of the location.

    Returns:
        float: Average temperature for the given season and location.
    """
    # Extract latitude and longitude from coordinates
    lat, lon = coords
    
    # Get the closest supported city
    closest_city = get_closest_city(lat, lon)
    
    # Define the months corresponding to each season
    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }

    # Get the months for the given season
    months = season_months.get(season.lower())
    if not months:
        raise ValueError("Invalid season provided")

    temperatures = []
    
    # Fetch temperature data for a random day in each month of the season
    for month in months:
        # Generate a random day within the month
        day = rng.integers(1, 28)  # Assuming 28 days to avoid issues with different month lengths
        date = datetime.date(2023, month, day).isoformat()
        
        try:        
            response = requests.get(
                closest_city['url'],
                params={
                    'refresh_count': 1,
                    'browser_zone': 'Eastern Daylight Time',
                    'date': date
                }
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            
            # Extract the temperature data
            if 'rows' in data:
                for row in data['rows']:
                    if 'c' in row and len(row['c']) > 1:
                        temp = row['c'][1]['v']
                        if temp is not None:
                            temperatures.append(temp)
            else:
                print(f"No temperature data available for date {date}: {data}")
        except requests.exceptions.RequestException:
            print(f"Temperature request failed for date {date}. Using locally saved temp data instead.")
            return get_temps_from_file(coords, seed, season)
        except ValueError:
            print(f"Error processing JSON response for date {date}. Using locally saved temp data instead.")
            return get_temps_from_file(coords, seed, season)
        except Exception:
            print(f"Generic error for temperature request on date {date}. Using locally saved temp data instead.")
            return get_temps_from_file(coords, seed, season)

    # Raise an exception if no temperature data was fetched
    if not temperatures:
        raise Exception("Failed to fetch temperature data")

    # Return the average temperature
    return int(sum(temperatures) / len(temperatures))

class EnvironmentClass:
    """
    Class representing the environment for EV routing and charging simulation.
    """

    def __init__(self, config_fname: str, seed: int, sub_seed: int, zone_coords: list, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize the environment with configuration, device, and dtype.

        Parameters:
            config_fname (str): Path to the configuration file.
            seed (int): Seed for getting the temp
            sub_seed (int): Seed for random number generator.
            zone_coords (list): Coordinates of the center of the zone.
            device (torch.device): Device to run tensor operations (CPU or CUDA).
            dtype (torch.dtype): Data type for tensors.
        """
        self.device = device
        self.dtype = dtype

        # Load configuration parameters for the environment
        config = load_config_file(config_fname)['environment_settings']

        # Seeding environment random generator
        rng = np.random.default_rng(sub_seed)

        self.temperature = get_temperature(config['season'], zone_coords, rng, seed)

        self.init_ev_info(config, self.temperature, rng)

        # Store environment parameters
        self.num_cars = config['num_of_cars']
        self.num_chargers = config['num_of_chargers']
        self.step_size = config['step_size']
        self.decrease_rates = torch.tensor(self.info['usage_per_hour'] / 70)
        self.increase_rate = config['increase_rate'] / 60
        self.max_steps = config['max_sim_steps']
        self.max_mini_steps = config['max_mini_sim_steps']
        self.state_dim = (self.num_chargers * 3 * 2) + 5
        self.charging_status = np.zeros(self.num_cars)
        self.historical_charges_needed = []

    def init_ev_info(self, config: dict, temperature: float, rng: np.random.Generator):
        """
        Initialize electric vehicle (EV) information based on the configuration.

        Parameters:
            config (dict): Configuration dictionary.
            temperature (float): Temperature of the zone.
            rng (np.random.Generator): Random number generator.
        """
        # Generating a random model
        model_indices = rng.integers(len(config['models']), size=config['num_of_cars'])

        # Using the indices to select the model type and corresponding configurations
        model_type = np.array([config['models'][index] for index in model_indices], dtype=str)
        usage_per_hour = np.array([config['usage_per_hour'][index] for index in model_indices], dtype=float)
        max_charge = np.array([config['max_charge'][index] for index in model_indices], dtype=int)

        # Based on https://www.mdpi.com/2032-6653/12/3/115
        # Efficiency drops significantly below 0°C, gradually above it
        if temperature < 0:
            temp_efficiency = 0.5 # 50% efficiency at freezing
        elif 0 <= temperature <= 25:
            temp_efficiency =  1 - (25 - temperature) * 0.02 # 2% efficiency drop per degree below 25°C
        else:
            temp_efficiency = 1 # Full efficiency at or above 25°C

        # Modify usage_per_hour based on the efficiency factor and convert back to int
        usage_per_hour = (usage_per_hour * temp_efficiency).astype(int)

        # Random starting charge between 0.5-x%, where x scales between 1-25% as sessions continue
        starting_charge = config['starting_charge'] + 2000 * (rng.random(config['num_of_cars']) - 0.5)

        # Defining a structured array
        dtypes = [('starting_charge', float),
                  ('max_charge', int),
                  ('usage_per_hour', int),
                  ('model_type', 'U50'),  # Adjust string length as needed
                  ('model_indices', int),
                  ('episode_starting_charge', float)]
        info = np.zeros(config['num_of_cars'], dtype=dtypes)

        # Store EVs information
        info['max_charge'] = max_charge
        info['model_type'] = model_type
        info['usage_per_hour'] = usage_per_hour
        info['model_indices'] = model_indices
        info['starting_charge'] = starting_charge
        info['episode_starting_charge'] = starting_charge
        self.info = info

    def get_ev_info(self) -> np.ndarray:
        """
        Get the electric vehicle (EV) information.

        Returns:
            np.ndarray: Array containing EV information.
        """
        return self.info

    def init_data(self):
        """
        Initialize data required for simulating the EV routing and charging process.

        Sets various tensors and arrays for simulation including tokens, destinations,
        capacity, stops, and battery levels.
        """
        starting_charge_array = np.array(self.info['starting_charge'], copy=True)
        starting_battery_level = torch.tensor(starting_charge_array, dtype=self.dtype, device=self.device)

        tokens = torch.tensor([[o_lat, o_lon] for (o_lat, o_lon, d_lat, d_lon) in self.routes], device=self.device)

        destinations = np.array([[d_lat, d_lon] for (o_lat, o_lon, d_lat, d_lon) in self.routes])
        destinations = torch.tensor(destinations, dtype=self.dtype, device=self.device)

        stops = torch.zeros((destinations.shape[0], max(len(path) for path in self.paths) + 1),\
                            dtype=self.dtype, device=self.device)
        target_battery_level = torch.zeros_like(stops, device=self.device)

        charging_stations = []
        station_ids = []

        for agent_index, path in enumerate(self.paths):
            prev_step = self.charges_needed[agent_index].shape[0] - 2

            if len(path) == 0:  # There are no stops to charge at
                stops[agent_index][0] = agent_index + 1
                target_battery_level[agent_index, 0] = self.charges_needed[agent_index][-2, -1]

                # Zero out values after the current step
                stops[agent_index, 1:] = 0
                target_battery_level[agent_index, 1:] = 0
            else:
                for step_index in range(len(stops[agent_index])):
                    if step_index > len(path):
                        break
                    elif step_index == len(path):  # Go to final destination
                        stops[agent_index][step_index] = agent_index + 1
                        target_battery_level[agent_index, step_index] = self.charges_needed[agent_index][prev_step, -1]
                    else:  # Go to stop
                        charger_id = self.unique_chargers[path[step_index]][0]

                        try:
                            station_index = station_ids.index(charger_id)
                            station_index += destinations.shape[0] + 1

                        except ValueError:  # Station not in list, so create a new station
                            station_ids.append(charger_id)
                            station_index = len(station_ids) + destinations.shape[0]
                            # Lat and long of charging station
                            stop = [self.unique_chargers[path[step_index]][1], self.unique_chargers[path[step_index]][2]]                        
                            charging_stations.append(stop)

                        stops[agent_index][step_index] = station_index
                        target_battery_level[agent_index][step_index] = self.charges_needed[agent_index][prev_step][self.local_paths[agent_index][step_index]]
                        prev_step = self.local_paths[agent_index][step_index]

        target_battery_level = target_battery_level[:, 1:]  # Ignore the battery it takes to get from the origin to the first stop
        charging_stations = np.array(charging_stations)

        if len(charging_stations) != 0:
            destinations = torch.vstack((destinations, torch.tensor(charging_stations, dtype=self.dtype, device=self.device)))

        capacity = torch.ones(len(charging_stations), dtype=self.dtype, device=self.device) * 10  # Dummy capacity of 10 cars for every station

        actions = torch.zeros((tokens.shape[0], destinations.shape[0]), device=self.device)
        move = torch.ones(tokens.shape[0], device=self.device)
        traffic = np.zeros(destinations.shape[0])

        # Storing in class
        self.move = move
        self.stops = stops
        self.tokens = tokens
        self.actions = actions
        self.capacity = capacity
        self.destinations = destinations
        self.target_battery_level = target_battery_level
        self.starting_battery_level = starting_battery_level

    def simulate_routes(self):
        """
        Simulate the environment for a matrix of tokens (vehicles) as they move towards their destinations,
        update their battery levels, and interact with charging stations.

        Returns:
            None
        """

        # Initialize routing data
        self.init_data()

        tokens = self.tokens
        battery = self.starting_battery_level
        destinations = self.destinations
        actions = self.actions
        moving = self.move
        target_battery_level = self.target_battery_level

        if target_battery_level.size(1) == 0:
            target_battery_level = torch.zeros(target_battery_level.size(0)).unsqueeze(1)

        stops = self.stops

        # Pre-process capacity array
        capacity = torch.cat((torch.zeros(tokens.shape[0], device=self.device), self.capacity))

        mini_step_count = 0
        tokens_size = tokens.shape
        paths = torch.empty((0, tokens_size[0], tokens_size[1]))
        traffic_per_charger = torch.empty((0, destinations.shape[0]))
        battery_levels = torch.empty((0, battery.shape[0]))
        distances_per_car = torch.zeros(1, tokens.shape[0])

        energy_used = np.zeros(self.num_cars)  # Initialize energy used for each car

        traffic_level = None

        done = False

        while (not done) and (mini_step_count <= self.max_mini_steps):
            stops[:, 0] -= 1

            # Get NxM matrix of actions
            actions = get_actions(actions, stops, dtype=self.dtype)

            # Move the tokens and get the updated position
            tokens, distance_travelled = move_tokens(tokens, moving, actions, destinations, self.step_size)

            # Track token position at each timestep and how far they traveled
            paths = torch.cat([paths, tokens.cpu().unsqueeze(0)], dim=0)
            distances_per_car = torch.cat([distances_per_car, distance_travelled.cpu().unsqueeze(0) +\
                                           distances_per_car[-1, :]], dim=0)

            # Get Nx1 matrix of distances
            distances = get_distance(tokens, destinations, actions)

            # Get Nx1 matrix of 0s or 1s that indicate if a car has arrived at current stop
            arrived = get_arrived(distances, self.step_size)

            # Accumulate traffic level of each station as Mx1 matrix
            traffic_level = get_traffic(stops, destinations, arrived)

            # Track traffic for each timestep
            traffic_per_charger = torch.cat([traffic_per_charger, traffic_level.cpu().unsqueeze(0)], dim=0)

            # Get charging or discharging rate for each car as Nx1 matrix
            charging_rates = get_charging_rates(stops, traffic_level, arrived, capacity, self.decrease_rates,\
                                                self.increase_rate, self.dtype)

            # Track which cars have reached their final destination
            distance_from_final = tokens - destinations[:len(tokens)]
            arrived_at_final = (distance_from_final[:, 0] == 0) & (distance_from_final[:, 1] == 0).int().unsqueeze(0)

            # Update the battery level of each car
            battery = update_battery(battery, charging_rates, arrived_at_final)
            battery_levels = torch.cat([battery_levels, battery.cpu().unsqueeze(0)], dim=0)

            # Track energy used (absolute value of charging rates)
            energy_used += torch.abs(charging_rates).cpu().numpy()

            # Check if the car is at their target battery level
            battery_charged = get_battery_charged(battery, target_battery_level, self.device)

            # Charging but ready to leave
            ready_to_leave = battery_charged * arrived

            # Charging and not ready to leave
            charging_status = arrived - ready_to_leave

            if DEBUG:
                print(f"STOPS:\n{stops}")
                print(f"TOKENS:\n{tokens}")
                print(f"DESTINATIONS:\n{destinations}")
                print(f"BATTERY:\n{battery}")
                print(f"TARGET BATTERY:\n{target_battery_level}")

            if torch.any(battery <= 0):
                # Print the graph for the car that ran out of battery
                negative_index = torch.where(battery <= 0)[0][0].item()
                print(f"\n\n---\n\nCharge graph of {negative_index} who died, mini-step {mini_step_count}:\n{self.charges_needed[negative_index]}")

                print(f"\n\n---\n\nHistorical charge graphs:")
                for row in self.historical_charges_needed:
                    # Use slicing to print every X-th column
                    print(row[negative_index])

                raise Exception("NEGATIVE BATTERY!")

            # Update which cars will move
            moving = (charging_status - 1) * -1

            # Zero-out tokens that are already at their stop
            diag_matrix = torch.diag(torch.tensor([0 if x == -1 else 1 for x in stops[:, 0]],\
                                                  dtype=self.dtype, device=stops.device))
            moving = torch.matmul(moving.to(self.dtype), diag_matrix)

            stops[:, 0] += 1

            # Change the stops array to shift over the next stop if the token is ready to leave
            stops = update_stops(stops, ready_to_leave, self.dtype, self.device)
            target_battery_level = update_stops(target_battery_level, ready_to_leave, self.dtype, self.device)

            # Increase step count
            mini_step_count += 1
            
            if min(arrived_at_final[0, :]) == 1:
                done = True

        
        # Calculate reward as -(distance * 100 + peak traffic + energy used / 100)
        distance_factor = distances_per_car[-1].numpy() * 100
        peak_traffic = np.max(traffic_per_charger.numpy())
        energy_used = energy_used / 100
        
        # Note that by doing (* 100) and (/ 100) we are scaling each factor of the reward to be around 0-10 on average
        
        self.simulation_reward = -(distance_factor + peak_traffic + energy_used)

        # Save results in class
        self.tokens = tokens
        self.new_starting_battery = battery
        self.charging_status = charging_status #still_status: 1 if car is still charging and 0 if not
        self.path_results = paths.numpy()
        self.traffic_results = traffic_per_charger.numpy()
        self.battery_levels_results = battery_levels.numpy()
        self.distances_results = distances_per_car.numpy()
        self.arrived_at_final = arrived_at_final

        return done

    def get_odt_info(self):
        return self.arrived_at_final[0,:]

    def get_results(self) -> tuple:
        """
        Get the results of the simulation.

        Returns:
            tuple: A tuple containing:
                - paths (list): List of token positions at each timestep.
                - traffic_per_charger (torch.Tensor): Tensor of traffic levels at each charging station over time.
                - battery_levels (list): List of battery levels at each timestep.
                - distances_per_car (list): List of distances traveled by each token at each timestep.
                - simulats (float): Reward for the simulation.
        """
        return self.path_results, self.traffic_results, self.battery_levels_results, self.distances_results,\
                self.simulation_reward

    def generate_paths(self, distribution: np.ndarray, fixed_attributes: list, agent_index: int):
        """
        Generate paths for the agents based on distribution and fixed attributes.

        Parameters:
            distribution (np.ndarray): Distribution array for generating paths.
            fixed_attributes (list): Fixed attributes for path generation.
        """
        # Generate graph of possible paths from chargers to each other, the origin, and destination
        graph = build_graph(self.agent.idx, self.step_size, self.info, self.agent.unique_chargers,\
                            self.agent.org_lat, self.agent.org_long, self.agent.dest_lat, self.agent.dest_long,\
                            self.charging_status[agent_index])
        self.charges_needed.append(copy.deepcopy(graph))

        if DEBUG:
            print("-------------")
            print(f"{agent_index} - CHARGES NEEDED - {graph}")

        # Redefine weights in graph
        for v in range(graph.shape[0] - 2):
            # Get multipliers from neural network
            if not fixed_attributes:
                traffic_mult = 1 - distribution[v]
                distance_mult = distribution[v]
            else:
                traffic_mult = fixed_attributes[0]
                distance_mult = fixed_attributes[1]

            # Distance * distance_mult + Traffic * traffic_mult
            graph[:, v] = graph[:, v] * distance_mult + self.agent.unique_traffic[v, 1] * traffic_mult

        path = dijkstra(graph, self.agent.idx)

        if DEBUG:
            print(f"{agent_index} - PATH - {path}")

        self.local_paths.append(copy.deepcopy(path))

        # Get stop ids from global list instead of only local to agent
        stop_ids = np.array([self.agent.unique_traffic[step, 0] for step in path])

        # Create a dictionary to map stop ids to their indices in self.traffic[:, 0]
        traffic_dict = {stop_id: idx for idx, stop_id in enumerate(self.traffic[:, 0])}

        # Create global_paths by preserving the order from the original path
        global_paths = np.array([traffic_dict[stop_id] for stop_id in stop_ids if stop_id in traffic_dict])

        self.paths.append(global_paths)

        # Update traffic
        for step in global_paths:
            self.traffic[step, 1] += 1

    def reset_agent(self, agent_idx: int, is_odt=False, is_madt=False) -> np.ndarray:
        """
        Reset the agent for a new simulation run.

        Parameters:
            agent_idx (int): Index of the agent to reset.
        
        Returns:
            np.ndarray: State array for the agent.
        """
        if is_odt or is_madt:
            agent_chargers = self.chargers[ 0, agent_idx, :]
        else:
            agent_chargers = self.chargers[agent_idx, :, 0]
        agent_unique_chargers = [charger for charger in self.unique_chargers if charger[0] in agent_chargers]
        agent_unique_traffic = np.array([[t[0], t[1]] for t in self.traffic if t[0] in agent_chargers])

        # Get distances from origin to each charging station
        org_lat, org_long, dest_lat, dest_long = self.routes[agent_idx]
        dists = np.array([haversine(org_lat, org_long, charge_lat, charge_long) for (id, charge_lat, charge_long) in agent_unique_chargers])
        route_dist = haversine(org_lat, org_long, dest_lat, dest_long)

        if is_madt:
            local_state = np.hstack((
                np.vstack((agent_unique_traffic[:, 1], dists)).reshape(-1),  # Traffic levels and distances to chargers
                np.array([route_dist]),  # Distance to the final destination
                np.array([self.info['model_indices'][agent_idx]])  # Model index of the agent's car
            ))
            
            # Global state
            global_state = np.hstack((
                np.array([self.num_chargers * 3]),  # Total number of chargers
                np.array([self.num_cars])  # Total number of cars
            ))
            self.agent = agent_info(agent_idx, agent_chargers, self.routes[agent_idx],
                                    agent_unique_chargers, agent_unique_traffic)
            return global_state, local_state

        # Traffic level and distance of each station plus total charger num, total distance,
        # number of EVs, car model index, and temperature
        state = np.hstack((np.vstack((agent_unique_traffic[:, 1], dists)).reshape(-1),
                           np.array([self.num_chargers * 3]), np.array([route_dist]),
                           np.array([self.num_cars]), np.array([self.info['model_indices'][agent_idx]]),
                           np.array([self.temperature])))

        # Storing agent info
        self.agent = agent_info(agent_idx, agent_chargers, self.routes[agent_idx],
                                agent_unique_chargers, agent_unique_traffic)
        return state

    def init_routing(self):
        # Clearing paths
        self.paths = []
        self.historical_charges_needed.append(self.charges_needed)
        self.charges_needed = []
        self.local_paths = []

        # Update starting routes
        if self.tokens != None:
            # Convert new_starting_positions tensor to a Python list
            new_starting_positions = self.tokens.tolist()
            
            # Iterate through each route and update the starting positions
            new_routes = []
            for i, route in enumerate(self.routes):
                if i < len(new_starting_positions):
                    # Update starting latitude and longitude
                    new_start_lat = new_starting_positions[i][0]
                    new_start_lon = new_starting_positions[i][1]
                    updated_route = [new_start_lat, new_start_lon, route[2], route[3]]
                    new_routes.append(updated_route)
                else:
                    # If there are more routes than positions, keep the original route
                    new_routes.append(route)
    
            # Update self.routes with the new starting positions
            self.routes = new_routes

            # Sets starting battery to ending battery of last timestep
            self.info['starting_charge'] = self.new_starting_battery.cpu()


    def reset_episode(self, chargers: np.ndarray, routes: np.ndarray, unique_chargers: np.ndarray):
        """
        Reset the episode with new chargers, routes, and unique chargers.

        Parameters:
            chargers (np.ndarray): Array of chargers.
            routes (np.ndarray): Array of routes.
            unique_chargers (np.ndarray): Array of unique chargers.
        """
        self.paths = []
        self.charges_needed = []
        self.historical_charges_needed = []
        self.local_paths = []
        self.tokens = None

        traffic = np.zeros(shape=(unique_chargers.shape[0], 2))
        traffic[:, 0] = unique_chargers['id']

        self.info['starting_charge'] = self.info['episode_starting_charge']  # Reset battery back to base level

        self.traffic = traffic  # [[charger id, traffic_leve],...]
        self.unique_chargers = unique_chargers  # [(charger id, charger latitude, charger longitude),...]
        self.chargers = chargers  # [[[charger id, charger latitude, charger longitude],...],...] (chargers[agent index][charger index][charger property index])
        self.routes = routes  # [[starting latitude, starting longitude, ending latitude, ending longitude],...]

    def cma_store(self):
        self.store_paths = copy.deepcopy(self.paths)
        self.store_charges_needed = copy.deepcopy(self.charges_needed)
        self.store_local_paths = copy.deepcopy(self.local_paths)

    def cma_copy_store(self):
        self.paths = copy.deepcopy(self.store_paths)
        self.charges_needed = copy.deepcopy(self.store_charges_needed)
        self.local_paths = copy.deepcopy(self.store_local_paths)

    def cma_clean(self):
        self.paths = copy.deepcopy(self.store_paths)
        self.charges_needed = copy.deepcopy(self.store_charges_needed)
        self.local_paths = copy.deepcopy(self.store_local_paths)
        
        self.store_paths = []
        self.store_charges_needed = []
        self.store_local_paths = []

if __name__ == "__main__":
    seeds = [1234, 5555, 2020, 5678, 9101]
    coords_list = [[43.02120034946083, -81.28349087468504],
                   [43.004969336049854, -81.18631870502043],
                   [42.95923445066671, -81.26016049362336],
                   [42.98111190139387, -81.30953935839466],
                   [42.9819404397449, -81.2508736429095]]
    
    save_temps(coords_list, seeds)