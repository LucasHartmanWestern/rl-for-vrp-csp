import json
import pandas as pd
import yaml
from geopy.geocoders import Nominatim
from geopy import Point
from geopy.distance import geodesic
import numpy as np

# Used to get the coordinates of the chargers from the API dataset
def get_charger_data():
    # From JSON file
    with open('data/Ontario_Charger_Dataset.json') as file:
        data = json.load(file)
    charger_data = []
    for station in [item for item in data['fuel_stations'] if item['city'] == 'London']:
        charger_id = station['id']
        charger_lat = round(station['latitude'], 5)
        charger_long = round(station['longitude'], 5)
        charger_data.append([charger_id, charger_lat, charger_long])
    charger_info = pd.DataFrame(charger_data, columns=['id', 'latitude', 'longitude'])

    return charger_info

# Get list of charger coordinates relative to where the origin, destination, and midpoint are
def get_charger_list(chargers, org_lat, org_long, dest_lat, dest_long, num_of_chargers):
    list_of_chargers = [(i, lat, long) for i, (lat, long) in enumerate(chargers)]

    # Calculate the midway point between origin and destination
    midway_lat = (org_lat + dest_lat) / 2
    midway_long = (org_long + dest_long) / 2

    # Get list of chargers around origin, destination, and midway point
    org_chargers = get_closest_chargers(org_lat, org_long, num_of_chargers, list_of_chargers)
    dest_chargers = get_closest_chargers(dest_lat, dest_long, num_of_chargers, list_of_chargers)
    midway_chargers = get_closest_chargers(midway_lat, midway_long, num_of_chargers, list_of_chargers)

    # Combine list
    combined_list = org_chargers + dest_chargers + midway_chargers

    # Remove duplicates and pad list so that there's always num_of_chargers * 3
    unique_list = list(set(combined_list))
    padding_chargers = get_closest_chargers(midway_lat, midway_long, num_of_chargers, list_of_chargers, unique_list)

    # Combine and append lists
    return unique_list + padding_chargers


def load_config_file(fname=None):
    # Created by Santiago 26/04/2024
    ''' Helper method to load the configuration parameters from yaml file
        first the default yaml file is loaded and the upadted with the
        new most updated configuration yaml file given by fname.
        args:
            fname: string-> path to the yaml config file
        output:
            parameters: python dict
    '''

    with open(fname) as config:
        parameters = yaml.safe_load(config)

    return parameters

def get_closest_chargers(current_lat, current_long, return_num, charger_list, existing_list=None):
    """Returns a list of chargering stations which are closest to the current coordinates

    Args:
        current_lat: Latitude to use as the origin
        current_long: Longitude to use as the origin
        return_num: Amount of charging stations to return
        charger_list: A list of tuples (id, lat, long)

    Returns:
        A filtered list of tuples (id, lat, long)
    """
    # Calculate the distances and travel times between the current coordinates and all charger locations
    distances_and_times = []
    origin = (current_lat, current_long)
    for charger in charger_list:
        destination = (charger[1], charger[2])
        distance, time = get_distance_and_time(origin, destination)
        distances_and_times.append((charger[0], distance, time))

    # Sort the distances in ascending order
    distances_and_times.sort(key=lambda x: x[1])

    # Get the closest charging stations up to the desired return_num
    closest_chargers = []

    if existing_list is None:
        for i in range(min(return_num, len(distances_and_times))):
            charger_id = distances_and_times[i][0]
            charger_lat = charger_list[charger_id][1]
            charger_long = charger_list[charger_id][2]
            closest_chargers.append((charger_id, charger_lat, charger_long))

    else: # Need more chargers because there was duplicates
        counter = 0
        i = return_num - 1
        while counter + len(existing_list) < return_num * 3:
            i += 1
            charger_id = distances_and_times[i][0]
            charger_lat = charger_list[charger_id][1]
            charger_long = charger_list[charger_id][2]
            if (charger_id, charger_lat, charger_long) in existing_list:
                continue
            else:
                closest_chargers.append((charger_id, charger_lat, charger_long))
                counter += 1

    return closest_chargers

def get_distance_and_time(origin, destination):
    """Gets the distance and travel-time between two coordinates using the OpenStreetMap's data"""

    # Assume average driving speed of 60 km/h
    average_speed = 60

    # Calculate the distance between origin and destination using geodesic distance
    distance = geodesic(origin, destination).km

    round(distance, 3)

    # Calculate the travel time based on the distance and average speed
    # Time = distance / speed
    time = round((distance / average_speed) * 3600, 3)  # time in seconds

    return distance, time

def get_org_dest_coords(center, radius, org_angle):
    lat, long = center
    km_in_degrees = radius / 111.11  # Approximation of km to degrees

    # Radius for origin point equals to the maximum radius
    org_radius = km_in_degrees

    # Calculate coordinates of origin point on the circle's circumference
    org_lat = lat + org_radius * np.sin(org_angle)
    org_long = long + org_radius * np.cos(org_angle)

    # Angle in radians for destination point
    dest_angle = (org_angle + np.pi) % (2 * np.pi)  # add π radians (180 degrees) to get the opposite point

    # Radius for destination point equals to the maximum radius
    dest_radius = km_in_degrees

    # Calculate coordinates of destination point on the circle's circumference
    dest_lat = lat + dest_radius * np.sin(dest_angle)
    dest_long = long + dest_radius * np.cos(dest_angle)

    # Generating an array of origin coordinates and dest cordinates
    # Rounded to 4 decimals for grid discretization
    orig_dest_coord = np.round(np.array([org_lat, org_long, dest_lat, dest_long]), decimals=4)

    return orig_dest_coord.transpose()
