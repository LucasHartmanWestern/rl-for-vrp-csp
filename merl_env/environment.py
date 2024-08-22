# Created by Lucas
# Restructured by Santiago 03/07/2024

import torch
import numpy as np
import copy

from ._helpers_routing import *
from ._pathfinding import dijkstra, build_graph, haversine
from .agent_info import agent_info

from data_loader import load_config_file

class EnvironmentClass:
    """
    Class representing the environment for EV routing and charging simulation.
    """

    def __init__(self, config_fname: str, sub_seed: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize the environment with configuration, device, and dtype.

        Parameters:
            config_fname (str): Path to the configuration file.
            sub_seed (int): Seed for random number generator.
            device (torch.device): Device to run tensor operations (CPU or CUDA).
            dtype (torch.dtype): Data type for tensors.
        """
        self.device = device
        self.dtype = dtype

        # Load configuration parameters for the environment
        config = load_config_file(config_fname)['environment_settings']

        # Seeding environment random generator
        rng = np.random.default_rng(sub_seed)

        self.init_ev_info(config, rng)

        # Store environment parameters
        self.num_cars = config['num_of_cars']
        self.num_chargers = config['num_of_chargers']
        self.step_size = config['step_size']
        self.decrease_rates = torch.tensor(self.info['usage_per_hour'] / 70)
        self.increase_rate = config['increase_rate'] / 60
        self.max_steps = config['max_sim_steps']
        self.max_mini_steps = config['max_mini_sim_steps']
        self.debug = config['debug']
        self.state_dim = (self.num_chargers * 3 * 2) + 4
        self.charging_status = np.zeros(self.num_cars)

        self.historical_charges_needed = []

    def init_ev_info(self, config: dict, rng: np.random.Generator):
        """
        Initialize electric vehicle (EV) information based on the configuration.

        Parameters:
            config (dict): Configuration dictionary.
            rng (np.random.Generator): Random number generator.
        """
        # Generating a random model
        model_indices = rng.integers(len(config['models']), size=config['num_of_cars'])

        # Using the indices to select the model type and corresponding configurations
        model_type = np.array([config['models'][index] for index in model_indices], dtype=str)
        usage_per_hour = np.array([config['usage_per_hour'][index] for index in model_indices], dtype=int)
        max_charge = np.array([config['max_charge'][index] for index in model_indices], dtype=int)

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

    def simulate_routes(self, timestep):
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

            # Check if the car is at their target battery level
            battery_charged = get_battery_charged(battery, target_battery_level, self.device)

            # Charging but ready to leave
            ready_to_leave = battery_charged * arrived

            # Charging and not ready to leave
            charging_status = arrived - ready_to_leave

            if self.debug:
                print(f"STOPS:\n{stops}")
                print(f"TOKENS:\n{tokens}")
                print(f"DESTINATIONS:\n{destinations}")
                print(f"BATTERY:\n{battery}")
                print(f"TARGET BATTERY:\n{target_battery_level}")

            if torch.any(battery <= 0):
                # Print the graph for the car that ran out of battery
                negative_index = torch.where(battery <= 0)[0][0].item()
                print(f"\n\n---\n\nCharge graph of {negative_index} who died on time-step {timestep + 1} mini-step {mini_step_count}:\n{self.charges_needed[negative_index]}")

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

        # Calculate reward as -(distance * 100 + peak traffic)
        self.simulation_reward = -(distances_per_car[-1].numpy() * 100 + np.max(traffic_per_charger.numpy()))

        # Save results in class
        self.tokens = tokens
        self.new_starting_battery = battery
        self.charging_status = charging_status #still_status: 1 if car is still charging and 0 if not
        self.path_results = paths.numpy()
        self.traffic_results = traffic_per_charger.numpy()
        self.battery_levels_results = battery_levels.numpy()
        self.distances_results = distances_per_car.numpy()
        

        return done, arrived_at_final

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
                            self.charging_status[agent_index], self.debug)
        self.charges_needed.append(copy.deepcopy(graph))

        if self.debug:
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

        if self.debug:
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

    def reset_agent(self, agent_idx: int, is_odt=False) -> np.ndarray:
        """
        Reset the agent for a new simulation run.

        Parameters:
            agent_idx (int): Index of the agent to reset.
        
        Returns:
            np.ndarray: State array for the agent.
        """
        if is_odt:
            agent_chargers = self.chargers[ 0,agent_idx, :]
        else:
            agent_chargers = self.chargers[agent_idx, :, 0]
        agent_unique_chargers = [charger for charger in self.unique_chargers if charger[0] in agent_chargers]
        agent_unique_traffic = np.array([[t[0], t[1]] for t in self.traffic if t[0] in agent_chargers])

        # Get distances from origin to each charging station
        org_lat, org_long, dest_lat, dest_long = self.routes[agent_idx]
        dists = np.array([haversine(org_lat, org_long, charge_lat, charge_long) for (id, charge_lat, charge_long) in agent_unique_chargers])
        route_dist = haversine(org_lat, org_long, dest_lat, dest_long)

        # Traffic level and distance of each station plus total charger num, total distance,
        # number of EVs, and car model index
        state = np.hstack((np.vstack((agent_unique_traffic[:, 1], dists)).reshape(-1),
                           np.array([self.num_chargers * 3]), np.array([route_dist]),
                           np.array([self.num_cars]), np.array([self.info['model_indices'][agent_idx]])))

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