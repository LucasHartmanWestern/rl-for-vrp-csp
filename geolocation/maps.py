import math
import requests
from geopy.distance import geodesic
import googlemaps

def travel_sim_full(origin, destination, travel_time, api_key):

    origin = address_to_coordinates(origin)
    destination = address_to_coordinates(destination)

    distance_details = get_distance_and_time(origin, destination)
    print(f"Starting Distance: {math.ceil(distance_details[0] / 1000)} KMs - {math.ceil(distance_details[1] / 60)} Minutes")

    i = 1
    repeat_multiplier = 1
    prev_location_list = []
    immediate_prev = None

    while get_distance_and_time(origin, destination)[1] > travel_time * repeat_multiplier:

        new_location = move_towards(origin, destination, travel_time * repeat_multiplier, api_key)

        if new_location == None:
            i += 1
            repeat_multiplier += 1
            continue

        immediate_prev = origin
        origin = new_location
        if origin in prev_location_list:
            i += 1
            repeat_multiplier += 1
            continue

        distance_details = get_distance_and_time(origin, destination)
        if isinstance(distance_details, str):
            i += 1
            repeat_multiplier += 1
            origin = immediate_prev
            continue

        repeat_multiplier = 1
        prev_location_list.append(origin)

        print(f"Interval {i}: {math.ceil(distance_details[0] / 1000)} KMs - {math.ceil(distance_details[1] / 60)} Minutes")
        i += 1

    print(f"\nArrived in {i} intervals travelling {travel_time / 60} minutes each interval!")

def get_distance_and_time(origin, destination, api_key):
    """Gets the distance and travel-time between two coordinates using the google maps API"""

    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=metric"
    # Format the tuples as strings
    origin = str(origin[0]) + "," + str(origin[1])
    destination = str(destination[0]) + "," + str(destination[1])

    r = requests.get(url + "&origins=" + origin + "&destinations=" + destination + "&key=" + api_key)
    data = r.json()

    if not data['rows']:
        return "No data found"

    if not data['rows'][0]['elements']:
        return "No elements in data found"

    element_data = data['rows'][0]['elements'][0]

    if 'distance' not in element_data or 'duration' not in element_data:
        return "Could not find distance or duration in element data"

    distance = element_data['distance']['value']
    time = element_data['duration']['value']

    return distance, time


def move_towards(origin, destination, travel_time, api_key):
    gmaps = googlemaps.Client(key=api_key)

    directions_result = gmaps.directions(origin, destination, mode="driving", departure_time="now")

    if not directions_result:
        return None

    shortest_route = min(directions_result, key=lambda route: route['legs'][0]['duration']['value'])
    route = shortest_route['legs'][0]
    total_duration = route['duration']['value']

    if total_duration < travel_time:
        return destination

    current_time = 0
    increment_time = 60  # Break down each step into 60 seconds increment
    for step in route['steps']:
        step_duration = step['duration']['value']

        for _ in range(step_duration // increment_time):
            current_time += increment_time
            if current_time >= travel_time:
                step_fraction = ((current_time - increment_time) + (step_duration % increment_time)) / step_duration

                start_lat = step['start_location']['lat']
                start_lng = step['start_location']['lng']
                end_lat = step['end_location']['lat']
                end_lng = step['end_location']['lng']

                start_point = (start_lat, start_lng)
                end_point = (end_lat, end_lng)
                bearing = get_initial_compass_bearing(start_point, end_point)
                distance = geodesic(start_point, end_point).miles
                new_point = geodesic(miles=distance * step_fraction).destination(point=start_point, bearing=bearing)
                new_lat, new_lng = new_point.latitude, new_point.longitude

                return (new_lat, new_lng)

    return None  # Unable to find a new point

def get_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formula used to calculate bearing is:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) – sin(lat1).cos(lat2).cos(Δlong))
    :param pointA: tuple of (latitude, longitude)
    :param pointB: tuple of (latitude, longitude)
    :returns: initial compass bearing in degrees, as float
    """

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2() returns values from -π to + π
    # so we need to normalize the result by converting it to a compass bearing
    # as it is common to measure bearings in this way.

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def address_to_coordinates(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(base_url, params=params)
    if response.status_code == 200 and response.json()['status'] == 'OK':
        latitude = response.json()['results'][0]['geometry']['location']['lat']
        longitude = response.json()['results'][0]['geometry']['location']['lng']
        return latitude, longitude
    else:
        return None

def get_closest_chargers(current_lat, current_long, return_num, charger_list):
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
    for i in range(min(return_num, len(distances_and_times))):
        charger_id = distances_and_times[i][0]
        charger_lat = charger_list[charger_id][1]
        charger_long = charger_list[charger_id][2]
        closest_chargers.append((charger_id, charger_lat, charger_long))

    return closest_chargers