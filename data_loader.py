import json
import pandas as pd
from geolocation.maps_free import get_closest_chargers, move_towards, get_distance_and_time

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