# Created by Lucas
# Restructured by Santiago 03/07/2024

import torch
import numpy as np
import copy

from ._helpers_routing import *
from ._pathfinding import *

from data_loader import load_config_file

class environment_class():

    def __init__(self, config_fname, sub_seed, device, dtype=torch.float32):
        """
        Parameters:
            device (torch.device): Cuda or cpu davice to work with tensors.
            dtype  (torch.dtype): dtype format to work with tensors.
        """
        self.device = device
        self.dtype  = dtype
        
        #loading configuration parameters for the environment
        config = load_config_file(config_fname)
        config = config['environment_settings']

        # Seeding environment random generator
        rng = np.random.default_rng(sub_seed)
        
        self.init_ev_info(config, rng)
      
        #store environment parameters
        self.num_of_agents = config['num_of_agents']
        self.num_of_chargers = config['num_of_chargers']
        self.step_size = config['step_size']
        self.decrease_rates = torch.Tensor(self.info['usage_per_hour'] / 60, device=device) 
        self.increase_rate = config['increase_rate'] / 60
        self.max_steps = config['max_sim_steps']

        
    def init_ev_info(self, c, rng):
        # Generating a random model
        model_indices = np.array([rng.integers(len(c['models'])) for agent in range(c['num_of_agents'])],\
                                 dtype=int)
        
        # Using the indices to select the model type and corresponding configurations
        model_type     = np.array([c['models'][index] for index in model_indices], dtype=str)
        usage_per_hour = np.array([c['usage_per_hour'][index] for index in model_indices], dtype=int)
        max_charge     = np.array([c['max_charge'][index] for index in model_indices], dtype=int)

        # Random charging between 0.5-x%, where x scales between 1-25% as sessions continue
        starting_charge = c['starting_charge'] + 2000*(rng.random(c['num_of_agents'])-0.5)

        # Defining a structured array
        dtypes = [('starting_charge', float),
                  ('max_charge', int),
                  ('usage_per_hour', int),
                  ('model_type', 'U50'),  # Adjust string length as needed
                  ('model_indices', int)]
        info = np.zeros(c['num_of_agents'], dtype=dtypes)

        # store EVs information
        info['max_charge'] = max_charge
        info['model_type'] = model_type
        info['usage_per_hour'] = usage_per_hour
        info['model_indices']  = model_indices
        info['starting_charge'] = starting_charge
        self.info = info
  
    def get_ev_info(self):
        return self.info

    def init_data(self):
        """
        Formats the data required for simulating the EV routing and charging process.

        Parameters:
            paths (list): List of paths for each EV.
            ev_routes (array): Array containing route information for each EV.
            unique_chargers (array): Array of unique charger locations.
            charges_needed (list): List of charging requirements for each EV.
            local_paths (list): List of local paths for each EV.

        Output self:
            - tokens (torch.tensor): Tensor of origin coordinates for each EV.
            - destinations (numpy.array): Array of destination coordinates, including charging stations.
            - capacity (torch.tensor): Tensor of capacity values for each destination.
            - stops (torch.tensor): Tensor indicating the stop sequence for each EV.
            - target_battery_level (numpy.array): Array of target battery levels at each stop.
            - starting_battery_level (torch.tensor): Tensor of initial battery levels for each EV.
            - actions (numpy.array): Array of actions for each EV.
            - move (numpy.array): Array indicating movement status for each EV.
            - traffic (numpy.array): Array indicating traffic levels at each destination.
        """

        starting_charge_array = np.array(self.info['starting_charge'], copy=True)
        starting_battery_level = torch.tensor(starting_charge_array, dtype=self.dtype, device=self.device) # 5000-7000

        tokens = torch.tensor([[o_lat, o_lon] for (o_lat, o_lon, d_lat, d_lon) in self.routes],\
                              device=self.device)

        destinations = np.array([[d_lat, d_lon] for (o_lat, o_lon, d_lat, d_lon) in self.routes])
        destinations = torch.tensor(destinations, dtype=self.dtype, device=self.device)

        stops = torch.zeros((destinations.shape[0], max(len(path) for path in self.paths) + 1),\
                            dtype=self.dtype, device=self.device)
        target_battery_level = torch.zeros_like(stops, device=self.device)

        # charging_stations = torch.zeros((len(paths),2), device=self.device)
        charging_stations = []
        station_ids = []

        for agent_index, path in enumerate(self.paths):

            prev_step = self.charges_needed[agent_index].shape[0] - 2

            for step_index in range(len(stops[agent_index])):

                if step_index == len(stops[agent_index]) - 1: # Go to final destination
                    stops[agent_index][step_index] = agent_index + 1
                    target_battery_level[agent_index, step_index] = self.charges_needed[agent_index][prev_step, -1]
                else:  # Go to stop
                    # Check if charger already exists in list
                    charger_id = self.unique_chargers[path[step_index]][0]

                    try:
                        station_index = station_ids.index(charger_id)
                        station_index = station_index + destinations.shape[0] + 1
                    except ValueError:  # Station not in list so create new station
                        station_ids.append(charger_id)
                        station_index = len(station_ids) + destinations.shape[0]
                        stop = [self.unique_chargers[path[step_index]][1], self.unique_chargers[path[step_index]][2]]  # Lat and long of charging station

                        # charging_stations[agent_index] = stop
                        charging_stations.append(stop)

                    stops[agent_index][step_index] = station_index

                    target_battery_level[agent_index][step_index] = self.charges_needed[agent_index][prev_step][self.local_paths[agent_index][step_index]]
                    prev_step = self.local_paths[agent_index][step_index]

        target_battery_level = target_battery_level[:, 1:]

        # destinations = torch.vstack((destinations, charging_stations))
        charging_stations = np.array(charging_stations)
        destinations = torch.vstack((destinations, torch.tensor(charging_stations, dtype=self.dtype, \
                                                                device=self.device)))

        actions = torch.zeros((tokens.shape[0], destinations.shape[0]), device=self.device)
        move = torch.ones(tokens.shape[0], device=self.device)
        traffic = np.zeros(destinations.shape[0])

        capacity = torch.ones(len(charging_stations), dtype=self.dtype, device=self.device) * 10 # Dummy capacity of 10 cars for every station

        #storing in class
        self.move  = move
        self.stops = stops
        self.tokens  = tokens
        self.actions = actions
        self.capacity= capacity
        self.destinations = destinations
        self.target_battery_level = target_battery_level
        self.starting_battery_level = starting_battery_level

    
 
    def simulate_routes(self):
        """
        Simulates the environment for a matrix of tokens (vehicles) as they move towards their destinations,
        update their battery levels, and interact with charging stations.

        Parameters:

        Returns:
            tuple: A tuple containing:
                - paths (list): List of token positions at each timestep.
                - traffic_per_charger (torch.Tensor): Tensor of traffic levels at each charging station over time.
                - battery_levels (list): List of battery levels at each timestep.
                - distances_per_car (list): List of distances traveled by each token at each timestep.
        """
        #Initialize routing data
        self.init_data()
        
        tokens  = self.tokens
        battery = self.starting_battery_level
        destinations = self.destinations
        actions = self.actions
        moving  = self.move
        target_battery_level = self.target_battery_level
        stops   = self.stops

        # Pre-process capacity array
        capacity = torch.concatenate((torch.zeros(tokens.shape[0],device=self.device), self.capacity))

        step_count = 0

        tokens_size = tokens.shape
        paths = torch.empty((0,tokens_size[0],tokens_size[1]))
        traffic_per_charger = torch.empty((0, destinations.shape[0]))
        battery_levels = torch.empty((0, battery.shape[0]))
        distances_per_car = torch.zeros(1,tokens.shape[0]) 


        init_stops = max(stops[:,0])
        while max(stops[:,0]) > 0 and step_count <= self.max_steps:

            stops[:, 0] -= 1

            # Get NxM matrix of actions
            actions = get_actions(actions, stops, dtype=self.dtype)

            # Move the tokens and get the update position
            tokens, distance_travelled = move_tokens(tokens, moving, actions, destinations, self.step_size)

            # Track token position at each timestep and how far they travelled
            paths = torch.cat([paths,tokens.cpu().unsqueeze(0)], dim=0)
            distances_per_car = torch.cat([distances_per_car, distance_travelled.cpu().unsqueeze(0) +\
                                           distances_per_car[-1,:]],dim=0)

            # Get Nx1 matrix of distances
            distances = get_distance(tokens, destinations, actions)

            # Get Nx1 matrix of 0s or 1s that indicate if a car has arrived at current stop
            arrived = get_arrived(distances, self.step_size)

            # Accumulate traffic level of each station as Mx1 matrix
            traffic_level = get_traffic(stops, destinations, arrived)

            # Track traffic for each timestep
            traffic_per_charger = torch.cat([traffic_per_charger, traffic_level.cpu().unsqueeze(0)], dim=0)

            # Get charging or discharging rate for each car as Nx1 matrix
            charging_rates = get_charging_rates(stops, traffic_level, arrived, capacity,\
                                                self.decrease_rates, self.increase_rate, self.dtype)

            # Update the battery level of each car
            battery = update_battery(battery, charging_rates)

            if torch.min(battery) < 0:
                print(distances)
                visualize_stats(traffic_per_charger, 'Change in Traffic Levels Over Time', 'Traffic Level')
                visualize_stats(battery_levels, 'Change in Battery Level Over Time', 'Battery Level')
                visualize_stats(distances_per_car, 'Distance Travelled Over Time', 'Distance Travelled')
                raise Exception(f"Battery level at {battery} - stepcount: {step_count}")

            battery_levels = torch.cat([battery_levels, battery.cpu().unsqueeze(0)], dim=0)

            # Check if the car is at their target battery level
            battery_charged = get_battery_charged(battery, target_battery_level)

            # Charging but ready to leave
            ready_to_leave = battery_charged * arrived

            # Charging and not ready to leave
            not_ready_to_leave = arrived - ready_to_leave

            # Update which cars will move
            moving = (not_ready_to_leave - 1) * -1

            # Zero-out tokens that are already at their stop
            diag_matrix = torch.diag(torch.tensor([0 if x == -1 else 1 for x in stops[:, 0]],\
                                                  dtype=self.dtype, device=stops.device))
            moving = torch.matmul(moving.to(self.dtype), diag_matrix)

            stops[:, 0] += 1

            # Change the stops array to shift over the next stop if the token is ready to leave
            stops = update_stops(stops, ready_to_leave, self.dtype)
            target_battery_level = update_stops(target_battery_level, ready_to_leave, self.dtype)

            # Increase step count
            step_count += 1

        # Calculate reward as -(distance * 100 + peak traffic)
        self.simulation_reward = -(distances_per_car[-1].numpy() * 100 + np.max(traffic_per_charger.numpy()))
        
        #saving results in class
        self.path_results = paths.numpy()
        self.traffic_results = traffic_per_charger.numpy()
        self.battery_levels_results = battery_levels.numpy()
        self.distances_results = distances_per_car.numpy()
    

    def get_results(self):
        """
        Parameters:

        Returns:
            tuple: A tuple containing:
                - paths (list): List of token positions at each timestep.
                - traffic_per_charger (torch.Tensor): Tensor of traffic levels at each charging station over time.
                - battery_levels (list): List of battery levels at each timestep.
                - distances_per_car (list): List of distances traveled by each token at each timestep.
                - simulation_rewards(list): 
        """

        return self.path_results, self.traffic_results, self.battery_levels_results,\
                self.distances_results, self.simulation_reward

    

    def generate_paths(self, distribution, fixed_attributes):

        ########### GENERATE GRAPH ###########

        # Build graph of possible paths from chargers to each other, the origin, and destination
        graph = build_graph(self.agent_idx, self.step_size, self.info, self.agent_unique_chargers,\
                            self.agent_org_lat, self.agent_org_long, self.agent_dest_lat, self.agent_dest_long)
        self.charges_needed.append(copy.deepcopy(graph))

        ########### REDEFINE WEIGHTS IN GRAPH ###########

        for v in range(graph.shape[0] - 2):
            # Get multipliers from neural network
            if not fixed_attributes:
                traffic_mult = 1 - distribution[v]
                distance_mult = distribution[v]
            else:
                traffic_mult = fixed_attributes[0]
                distance_mult = fixed_attributes[1]

            # Distance * distance_mult + Traffic * traffic_mult
            graph[:, v] = graph[:, v] * distance_mult + self.agent_unique_traffic[v, 1] * traffic_mult

        # Set last column to zero so long as it's not infinity for every row except the last 2
        mask = (graph[:-2, -1] != np.inf)
        graph[:-2, -1][mask] = 0

         ########### SOLVE WEIGHTED GRAPH ###########

        path = dijkstra(graph, self.agent_idx)

        self.local_paths.append(copy.deepcopy(path))

        # Get stop ids from global list instead of only local to agent
        stop_ids = np.array([self.agent_unique_traffic[step, 0] for step in path])
        global_paths = np.where(np.isin(self.traffic[:, 0], stop_ids))[0]

        self.paths.append(global_paths)

        ########### UPDATE TRAFFIC ###########
        for step in global_paths:
            self.traffic[step, 1] += 1

        
        
    def reset_agent(self, agent_idx):

        agent_chargers = self.chargers[agent_idx, :, 0]
        agent_unique_chargers = [charger for charger in self.unique_chargers if charger[0] in agent_chargers]
        agent_unique_traffic = np.array([[t[0], t[1]] for t in self.traffic if t[0] in agent_chargers])

        # Get distances from origin to each charging station
        org_lat, org_long, dest_lat, dest_long = self.routes[agent_idx]
        dists = np.array([haversine(org_lat, org_long, charge_lat, charge_long)\
                          for (id, charge_lat, charge_long) in agent_unique_chargers])
        route_dist = haversine(org_lat, org_long, dest_lat, dest_long)

        # Traffic level and distance of each station plus total charger num, total distance,
        # number of EVs, and car model index
        state = np.hstack((np.vstack((agent_unique_traffic[:, 1], dists)).reshape(-1),\
                           np.array([self.num_of_chargers * 3]), np.array([route_dist]),\
                           np.array([self.num_of_agents]), np.array([self.info['model_indices'][agent_idx]])))

        #storing agent info
        self.agent_idx = agent_idx
        self.agent_chargers =  agent_chargers
        self.agent_org_lat  = org_lat
        self.agent_org_long = org_long
        self.agent_dest_lat = dest_lat
        self.agent_dest_long= dest_long
        self.agent_unique_chargers = agent_unique_chargers
        self.agent_unique_traffic  = agent_unique_traffic
        
        return state
 
    def reset_episode(self, chargers, routes, unique_chargers):
        self.paths = []
        self.charges_needed = []
        self.local_paths = []
        
        traffic = np.zeros(shape=(unique_chargers.shape[0], 2))
        traffic[:, 0] = unique_chargers['id']
        
        self.traffic = traffic
        self.unique_chargers = unique_chargers
        self.chargers = chargers
        self.routes = routes