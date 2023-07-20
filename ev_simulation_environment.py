import copy
import csv
import math
import json
import random

import pandas as pd

from collections import OrderedDict
from geolocation.maps_free import get_closest_chargers, move_towards, get_distance_and_time
from geolocation.visualize import read_excel_data

from charging_station import ChargingStation

# TODO - Create simulation environment which is capable of:
#  - Introducing randomness (traffic will randomly fluctuate at charging stations based on time-of-day)
#  - Considering different makes and mmodels and decreasing (or increasing while charging) an E   Vs SoC propotionately
#  - Considering the load placed on each charging station relative to the traffic at the station
#  - Simulating the travel from an origin to destination

class EVSimEnvironment:
    def __init__(
            self,
            max_num_timesteps,
            num_of_episodes,
            num_of_chargers,
            make,
            model,
            cur_soc,
            max_soc,
            routes,
            seed=[0]
    ):
        """Create environment

        Args:
            num_of_chargers: Amount of chargers to consider
            make: Make of EV
            model: Model of EV
            cur_soc: Current state of Charge of EV in Watts
            max_soc: Maximum SoC capability in Watts
            org_lat: Latitude of route origin
            org_long: Longitude of route origin
            dest_lat: Latitude of route destination
            dest_long: Longitude of route destination

        Returns:
            Environment to use for EV simulations
        """

        self.seed = seed

        self.tracking_baseline = False

        self.agent_index = 0
        self.agent_list = [index for index in range(len(routes))]

        self.num_of_episodes = num_of_episodes
        self.episode_num = -1
        self.visited_list = []
        self.current_path = []
        self.best_path = []
        self.used_chargers = []
        self.charger_coords = []

        self.state = [() for route in routes]

        self.distances = [get_distance_and_time((route[0], route[1]), (route[2], route[3]))[0] for route in routes]

        self.charger_list = {} # List of ChargingStation objects

        self.step_num = [0 for route in routes]
        self.max_num_timesteps = max_num_timesteps
        self.episode_reward = [0 for route in routes]

        self.max_reward = math.inf * -1

        self.prev_charging = [None for route in routes]
        self.is_charging = [False for route in routes]

        self.num_of_chargers = num_of_chargers
        self.make = make
        self.model = model

        self.cur_soc = [cur_soc for route in routes]
        self.max_soc = max_soc
        self.base_soc = copy.copy(cur_soc) # Used to reset

        self.org_lat = [route[0] for route in routes]
        self.org_long = [route[1] for route in routes]
        self.dest_lat = [route[2] for route in routes]
        self.dest_long = [route[3] for route in routes]

        self.cur_lat = copy.copy(self.org_lat)
        self.cur_long = copy.copy(self.org_long)

        self.average_reward = []

        self.usage_per_hour = self.ev_info()

        self.get_charger_data()
        self.get_charger_list()

        for a in range(len(self.org_lat)):
            self.update_state(a)

    # Used to get the coordinates of the chargers from the API dataset
    def get_charger_data(self):

        # From JSON file
        with open('data/Ontario_Charger_Dataset.json') as file:
            data = json.load(file)
        charger_data = []
        for station in data['fuel_stations']:
            charger_id = station['id']
            charger_lat = round(station['latitude'], 4)
            charger_long = round(station['longitude'], 4)
            charger_data.append([charger_id, charger_lat, charger_long])
        self.charger_info = pd.DataFrame(charger_data, columns=['id', 'latitude', 'longitude'])

        self.charger_lat = self.charger_info.iloc[:, 1].tolist()  # Extract values from the 3rd column
        self.charger_long = self.charger_info.iloc[:, 2].tolist()  # Extract values from the 4th column

    # Step function of sim environment
    def step(self, action):
        """Simulate a step in the EVs travel

        Args:
            action: Number to use as index for stations list

        Returns:
            next_state: New state of the system
            reward: Reward for current state
            done: Indicator for if simulation is done
        """

        self.step_num[self.agent_list[self.agent_index]] += 1

        # Update traffic, SoC, and geographical position
        done = self.move(action)

        if self.agent_index == len(self.agent_list) - 1: # Only update the traffic after each agent had a turn
            self.update_traffic()

        self.update_charge(action)

        # Update state
        self.update_state()

        # Get reward of current state
        reward = self.reward(self.state, done)

        # Episode reward is the sum of the reward at each step
        self.episode_reward[self.agent_list[self.agent_index]] += reward

        # Log every tenth episode
        if self.episode_num % math.ceil(self.num_of_episodes / 10) == 0 or self.tracking_baseline:
            time_to_destination = get_distance_and_time((self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]]), (self.dest_lat[self.agent_list[self.agent_index]], self.dest_long[self.agent_list[self.agent_index]]))[1] / 60
            if time_to_destination <= 1 and self.cur_soc[self.agent_list[self.agent_index]] > 0 and done:
                self.log(action, True)
            else:
                self.log(action)

        # Check if using multiple agents or just 1
        if len(self.org_lat) > 1:

            state = copy.copy(self.state[self.agent_list[self.agent_index]])

            if done:
                del self.agent_list[self.agent_index]

                if len(self.agent_list) != 0:
                    self.agent_index = self.agent_index % len(self.agent_list)
            else:
                self.agent_index = (self.agent_index + 1) % len(self.agent_list) # Go to next agent now


            return state, reward, done
        else:
            return self.state[0], reward, done

    # Simulates battery life of EV as it travels
    def update_charge(self, action):

        charger_id = self.charger_coords[self.agent_list[self.agent_index]][action - 1][0]
        station = self.charger_list[charger_id]

        # Find how far station is away from current coordinates in minutes
        time_to_station = get_distance_and_time((self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]]), (station.coord[0], station.coord[1]))[1] / 60

        # Consume battery while driving
        if self.is_charging[self.agent_list[self.agent_index]] is not True or self.prev_charging[self.agent_list[self.agent_index]] != charger_id:
            self.cur_soc[self.agent_list[self.agent_index]] -= self.usage_per_hour / (60)

        # Increase battery while charging
        else:
            self.cur_soc[self.agent_list[self.agent_index]] += station.charge() / 60
            # Cap SoC at max
            if self.cur_soc[self.agent_list[self.agent_index]] > self.max_soc:
                self.cur_soc[self.agent_list[self.agent_index]] = self.max_soc

        # Start charging if within range of charging station
        if action != 0 and time_to_station <= 0.01:
            self.is_charging[self.agent_list[self.agent_index]] = True
            self.prev_charging[self.agent_list[self.agent_index]] = charger_id
        # Not at a station
        else:
            self.is_charging[self.agent_list[self.agent_index]] = False
            self.prev_charging[self.agent_list[self.agent_index]] = None


    # Simulates traffic updates at chargers
    def update_traffic(self):
        for charger in self.charger_list:
            self.charger_list[charger].update_traffic(self.seed + self.episode_num)

    # Simulates geographical movement of EV
    def move(self, action):
        # Find out how far EV can travel given current charge
        usage_per_hour = self.ev_info()
        max_distance = self.cur_soc[self.agent_list[self.agent_index]] / (usage_per_hour / 60)
        travel_distance = min(max_distance, 1)

        # Find how far destination is away from current coordinates in minutes
        time_to_destination = get_distance_and_time((self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]]), (self.dest_lat[self.agent_list[self.agent_index]], self.dest_long[self.agent_list[self.agent_index]]))[1] / 60

        # EV has reached destination or ran out of battery before reaching destination
        if time_to_destination < 1 or self.cur_soc[self.agent_list[self.agent_index]] <= 0 or self.step_num[self.agent_list[self.agent_index]] == self.max_num_timesteps:
            return True

        # EV is driving to destination
        if action == 0:
            # Drive towards selected destination
            self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]] = move_towards((self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]]), (self.dest_lat[self.agent_list[self.agent_index]], self.dest_long[self.agent_list[self.agent_index]]), travel_distance)

        # EV is driving towards a charging station
        else:
            # Drive towards station
            self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]] = move_towards((self.cur_lat[self.agent_list[self.agent_index]], self.cur_long[self.agent_list[self.agent_index]]), (self.charger_coords[self.agent_list[self.agent_index]][action - 1][1], self.charger_coords[self.agent_list[self.agent_index]][action - 1][2]), travel_distance)

        return False

    # Used to log the info for the paths
    def log(self, action, final = False, episode_offset = 0, a_index=None):
        new_row = []

        usage_per_hour = self.ev_info()

        if a_index is None:
            index = self.agent_list[self.agent_index]
        else:
            index = a_index

        # ID of path (Baseline or ep num)
        if self.tracking_baseline:
            new_row.append('Baseline')
        else:
            new_row.append(self.episode_num + episode_offset)

        new_row.append(index)

        # Action of path
        if action == -1:
            new_row.append('No Action')
        elif action == 0:
            new_row.append(action)
        else:
            new_row.append(self.charger_coords[index][action - 1][0])

        # Step number within episode
        new_row.append(self.step_num[index])

        # SoC in kW
        new_row.append(round(self.cur_soc[index] / 1000, 2))

        # EV is charging bool
        new_row.append(self.is_charging[index])

        # Episode reward (sum of step rewards up to this timestep)
        new_row.append(round(self.episode_reward[index], 2))

        # Coordinates (current or destination depending if episode is done)
        if final is not True:
            new_row.append(self.cur_lat[index])
            new_row.append(self.cur_long[index])
        else:
            new_row.append(self.dest_lat[index])
            new_row.append(self.dest_long[index])

        new_row.append(self.distances[index])

        max_time_on_soc = self.cur_soc[index] / (usage_per_hour / 60)
        time_left_in_trip = get_distance_and_time((self.cur_lat[index], self.cur_long[index]), (self.dest_lat[index], self.dest_long[index]))[1] / 60

        new_row.append(round(max_time_on_soc, 2))
        new_row.append(round(time_left_in_trip, 2))

        # Entire state (used for debugging)
        # new_row.append(self.state)

        if self.tracking_baseline is not True:
            self.current_path.append(new_row)
        self.visited_list.append(new_row)

    # Get list of charger coordinates relative to where the origin, destination, and midpoint are
    def get_charger_list(self):
        list_of_chargers = list(zip(self.charger_lat, self.charger_long))
        list_of_chargers = [(i, val1, val2) for i, (val1, val2) in enumerate(list_of_chargers)]

        # Calculate the midway point between origin and destination
        for i in range(len(self.org_lat)):
            midway_lat = (self.org_lat[i] + self.dest_lat[i]) / 2
            midway_long = (self.org_long[i] + self.dest_long[i]) / 2

            # Get list of chargers around origin, destination, and midway point
            org_chargers = get_closest_chargers(self.org_lat[i], self.org_long[i], self.num_of_chargers, list_of_chargers)
            dest_chargers = get_closest_chargers(self.dest_lat[i], self.dest_long[i], self.num_of_chargers, list_of_chargers)
            midway_chargers = get_closest_chargers(midway_lat, midway_long, self.num_of_chargers, list_of_chargers)

            # Combine and append lists
            self.charger_coords.append(org_chargers + dest_chargers + midway_chargers)

            # Create list of ChargingStation objects
            for charger in self.charger_coords[i]:
                if charger[0] not in self.charger_list:
                    self.charger_list[charger[0]] = ChargingStation(charger[0], (charger[1], charger[2]), len(self.is_charging))

        self.update_traffic()

        # Legacy code - not really useful anymore
        for charger in self.charger_coords[i]:
            self.used_chargers.append(charger)

    # Reset all states
    def reset(self):
        self.agent_list = [index for index in range(len(self.org_lat))]
        self.agent_index = 0

        if self.episode_num != -1: # Ignore initial reset

            # Track best path
            if self.episode_reward[self.agent_list[self.agent_index]] > self.max_reward and len(self.current_path) != 0:
                self.best_path = self.current_path.copy()
                self.max_reward = self.episode_reward[self.agent_list[self.agent_index]]

            if self.tracking_baseline is not True: # Ignore baseline in average calculations

                average_ep_reward = 0
                for agent in self.agent_list:
                    average_ep_reward += self.episode_reward[self.agent_list[agent]]
                average_ep_reward /= len(self.agent_list)

                # Track average reward of all episodes
                if self.episode_num == 0:
                    self.average_reward.append((average_ep_reward, 0))
                else:
                    prev_reward = self.average_reward[-1][0]
                    prev_reward *= self.episode_num
                    self.average_reward.append(((prev_reward + average_ep_reward) / (self.episode_num + 1), self.episode_num))

        # Reset all EVs to initial values
        for i in range(len(self.org_lat)):
            self.step_num[i] = 0
            self.episode_reward[i] = 0
            self.cur_soc[i] = self.base_soc
            self.cur_lat[i] = self.org_lat[i]
            self.cur_long[i] = self.org_long[i]

        self.current_path = []

        for charger in self.charger_list:
            self.charger_list[charger].reset()

        if self.tracking_baseline is not True: # Ignore basline in average calculations
            self.episode_num += 1

        # Log starting point on the sim graph (will always be the origin point hence the -1)
        if self.episode_num >= 0 or self.tracking_baseline:
            if (self.episode_num <= self.num_of_episodes and self.episode_num % math.ceil(self.num_of_episodes / 10) == 0) or self.tracking_baseline:
                for a_ind in range(len(self.org_lat)):
                    self.log(-1, False, 0, a_ind)

        # Update state
        self.update_state()

        return self.state[self.agent_list[self.agent_index]]

    # Scale negative rewards to fractions
    def reward(self, state, done):
        reward = 0
        charge_bool, battery_percentage, distance_to_dest, *charger_distances = state[self.agent_list[self.agent_index]]

        distance_from_origin, time_from_origin = get_distance_and_time((self.org_lat[self.agent_list[self.agent_index]], self.org_long[self.agent_list[self.agent_index]]), (self.dest_lat[self.agent_list[self.agent_index]], self.dest_long[self.agent_list[self.agent_index]]))

        # Find max traffic
        max_traffic = 0
        for charger in self.charger_list:
            charger_traffic = self.charger_list[charger].traffic
            if charger_traffic > max_traffic:
                max_traffic = charger_traffic

        # Reward negatively for high traffic at stations
        reward -= max_traffic

        # Decrease reward proportionately to distance remaining distance and battery percentage
        reward -= (distance_to_dest / distance_from_origin)

        return reward

    # TODO - Get EV Info
    def ev_info(self):
        # TODO - Make estimates more realistic using LH Dataset
        usage_per_hour = 15600 # Average usage per hour of Tesla
        return usage_per_hour

    def update_state(self, index=None):

        if index is None:
            index = self.agent_list[self.agent_index]

        charger_info = [] # Set of charger distance, current traffic, peak traffic, and charging rate
        for charger in self.charger_coords[index]: # Populate above set
            station = self.charger_list[charger[0]]
            charger_info.append(get_distance_and_time((self.cur_lat[index], self.cur_long[index]), (station.coord[0], station.coord[1]))[0])
            charger_info.append(station.traffic)

        # Recalculate remaining distance to destination
        total_dist = get_distance_and_time((self.org_lat[index], self.org_long[index]), (self.dest_lat[index], self.dest_long[index]))[0]

        if self.is_charging[index] is True:
            charge_bool = 1
        else:
            charge_bool = 0

        # Update state
        self.state[index] = (self.num_of_chargers, len(self.org_lat), total_dist, *charger_info)

    # Used for creating NNs
    def get_state_action_dimension(self):
        states = len(self.state[0])
        actions = 1 + (self.num_of_chargers * 3)
        return states, actions

    # Used for displaying paths on the graph sim
    def write_path_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            header_row = []
            header_row.append('Episode Num')
            header_row.append('Agent Num')
            header_row.append('Action')
            header_row.append('Timestep')
            header_row.append('SoC')
            header_row.append('Is Charging')
            header_row.append('Episode Reward')
            header_row.append('Latitude')
            header_row.append('Longitude')
            header_row.append('Distance')
            header_row.append('Max Time Left')
            header_row.append('Time to Destination')
            header_row.append('State')

            writer.writerow(header_row)

            for row in self.visited_list:
                writer.writerow(row)

            for row in self.best_path:
                row[0] = "Best"
                writer.writerow(row)

    # Used for displaying all chargers on graph sim
    def write_chargers_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Charger ID', 'Latitude', 'Longitude'])
            for charger in self.charger_list:
                writer.writerow((self.charger_list[charger].id, self.charger_list[charger].coord[0], self.charger_list[charger].coord[1]))

    def write_charger_traffic_to_csv(self, filepath):
        charger_traffic = []
        for charger in self.charger_list:
            charger_traffic.append(self.charger_list[charger].charge_statistics)

        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Charger ID', 'Timestep', 'Load', 'Traffic', 'Max Load'])
            for charger in charger_traffic:
                for timestep in charger:
                    writer.writerow(timestep)

    # Used for creating average reward graph
    def write_reward_graph_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average Reward', 'Episode Num'])

            for row in self.average_reward:
                writer.writerow(row)