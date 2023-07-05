import math
from geopy.geocoders import Nominatim
from geopy import Point
from geopy.distance import geodesic

def travel_sim_full(
        origin,
        destination,
        travel_time=1800):

    # Convert origin and destination addresses to coordinates
    origin = address_to_coordinates(origin)
    destination = address_to_coordinates(destination)

    # Get the initial distance and time between origin and destination
    distance_details = get_distance_and_time(origin, destination)
    print(f"Starting Distance: {math.ceil(distance_details[0])} KMs - {math.ceil(distance_details[1] / 60)} Minutes")

    i = 1
    repeat_multiplier = 1
    prev_location_list = []
    immediate_prev = None

    # Continue looping until the remaining travel time is less than the specified travel time
    while get_distance_and_time(origin, destination)[1] > travel_time * repeat_multiplier:

        # Move towards the destination based on the current travel time and multiplier
        new_location = move_towards(origin, destination, travel_time * repeat_multiplier)

        if new_location == None:
            # If moving towards the destination is not possible, increase the multiplier and continue the loop
            i += 1
            repeat_multiplier += 1
            continue

        immediate_prev = origin
        origin = new_location

        if origin in prev_location_list:
            # If the new location has already been visited, increase the multiplier and continue the loop
            i += 1
            repeat_multiplier += 1
            continue

        # Get the updated distance and time between the new origin and destination
        distance_details = get_distance_and_time(origin, destination)

        if isinstance(distance_details, str):
            # If there is an error in getting the distance and time, increase the multiplier, revert to the previous origin, and continue the loop
            i += 1
            repeat_multiplier += 1
            origin = immediate_prev
            continue

        # Reset the multiplier and add the new origin to the list of visited locations
        repeat_multiplier = 1
        prev_location_list.append(origin)

        # Print the current interval's distance and time
        print(f"Interval {i}: {math.ceil(distance_details[0])} KMs - {math.ceil(distance_details[1] / 60)} Minutes")
        i += 1

    # Print the final result
    print(f"\nArrived in {i} intervals travelling {travel_time / 60} minutes each interval!")

def get_distance_and_time(origin, destination):
    """Gets the distance and travel-time between two coordinates using the OpenStreetMap's data"""

    # Assume average driving speed of 60 km/h
    average_speed = 60

    # Calculate the distance between origin and destination using geodesic distance
    distance = geodesic(origin, destination).km

    # Calculate the travel time based on the distance and average speed
    # Time = distance / speed
    time = distance / average_speed * 3600  # time in seconds

    return distance, time

def move_towards(origin, destination, travel_time):
    # Assume average driving speed of 60 km/h
    average_speed = 60  # km/h

    # Calculate the travel distance based on the travel time and average speed
    travel_distance = (travel_time / 60) * average_speed  # distance = speed * time

    # Calculate total distance between origin and destination
    total_distance = geodesic(origin, destination).km

    if total_distance < travel_distance:
        # If the total distance is less than the travel distance, return the destination as the new location
        return destination

    # Calculate the bearing from origin to destination
    bearing = get_initial_compass_bearing(origin, destination)

    # Calculate the new point after moving towards the destination
    new_location = geodesic(kilometers=travel_distance).destination(Point(*origin), bearing)

    return new_location.latitude, new_location.longitude

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
        # Check if the input points are tuples
        raise TypeError("Only tuples are supported as arguments")

    # Convert latitude and longitude to radians
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    # Calculate the difference in longitude
    diffLong = math.radians(pointB[1] - pointA[1])

    # Calculate the bearing components
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    # Calculate the initial bearing using atan2
    initial_bearing = math.atan2(x, y)

    # Normalize the initial bearing to a compass bearing (0-360 degrees)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def address_to_coordinates(address):
    # Convert an address to coordinates using Nominatim geocoder
    geolocator = Nominatim(user_agent="datafev-sim")
    location = geolocator.geocode(address)
    return (location.latitude, location.longitude)

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