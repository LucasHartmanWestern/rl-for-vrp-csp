import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import csv
import ast
import torch

def generate_plot(df):

    """
    Visualizes the data using a graph. The graph connects the coordinates in the Latitude and Longitude columns
    and plots the coordinates in the various (C1_Latitude, C1_Longitude, ..., C10_Latitude, C10_Longitude) columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.

    Returns:
        None
    """

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Plot main path
    plt.plot(df['Longitude'], df['Latitude'], 'ro-', label='Main Path')

    # Plot additional paths
    for i in range(1, 11):
        lat_key = f'C{i}_Latitude'
        lon_key = f'C{i}_Longitude'
        if lat_key in df.columns and lon_key in df.columns:
            plt.plot(df[lon_key], df[lat_key], 'o-', label=f'Charger {i}')

    # Add legend, grid, title and labels
    plt.legend()
    plt.grid(True)
    plt.title('Paths')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()

def plot_aggregate_output_values_per_route(data):

    """
    Plots the average output values of neurons over time for each route index.

    Parameters:
        data (list): A nested list containing tuples of the form
                     (episode_avg_output_values, episode_number, aggregation_num, route_index, seed).

    Returns:
        None
    """

    # Extract unique route indices and aggregation numbers
    route_indices = sorted(set(item[3] for sublist in data for item in sublist))
    aggregation_nums = sorted(set(item[2] for sublist in data for item in sublist))
    num_neurons = len(data[0][0][0])  # Assuming all entries have the same number of neurons
    max_episodes = max(item[1] for sublist in data for item in sublist)  # Maximum number of episodes

    # Set up the plot for each route index
    for route_index in route_indices:
        plt.figure(figsize=(10, 6))

        # Plot each neuron as a separate line
        for neuron_index in range(num_neurons):
            # Calculate the time parameter for each data point
            time_and_values = [(item[1] + item[2] * max_episodes, item[0][neuron_index]) for sublist in data for item in sublist if item[3] == route_index]
            time, avg_output_values = zip(*sorted(time_and_values, key=lambda x: x[0]))
            plt.plot(time, avg_output_values, label=f'Neuron {neuron_index + 1}')

        # Plot vertical lines for each aggregation number
        for agg_num in aggregation_nums:
            plt.axvline(x=agg_num * max_episodes, color='grey', linestyle='--', label=f'Aggregation {agg_num}')

        # Set title, labels, and legend
        seed = data[0][0][4]  # Assuming all entries have the same seed
        plt.title(f'Average Output Values for Route Index: {route_index}, Seed: {seed}')
        plt.xlabel('Time')
        plt.ylabel('Average Output Value')
        plt.legend()

        # Show the plot
        plt.show()


def plot_aggregate_reward_data(data):

    """
    Plots the average rewards over time for each route index, with vertical lines indicating aggregation steps.

    Parameters:
        data (list): A nested list containing tuples of the form (avg_reward, aggregation_num, route_index, seed).

    Returns:
        None
    """

    # Extract unique route indices and aggregation numbers
    route_indices = sorted(set(item[2] for sublist in data for item in sublist))
    aggregation_nums = sorted(set(item[1] for sublist in data for item in sublist))

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each route index as a separate line
    for route_index in route_indices:
        avg_rewards = [item[0] for sublist in data for item in sublist if item[2] == route_index]
        plt.plot(range(len(avg_rewards)), avg_rewards, label=f'Route Index: {route_index}')

    # Plot vertical lines for each aggregation number
    for i, agg_num in enumerate(aggregation_nums):
        plt.axvline(x=i * len(data[0]), color='grey', linestyle='--', label=f'Aggregation {agg_num}' if i == 0 else None)

    # Set title, labels, and legend
    seed = data[0][0][3]  # Assuming all entries have the same seed
    plt.title(f'Average Reward for Seed: {seed}')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.legend()

    # Show the plot
    plt.show()

def generate_average_reward_plot(algorithm, df, session_number):

    """
    Visualizes the average reward over time per episode using a graph.

    Parameters:
        algorithm (str): The name of the algorithm used for training.
        df (pandas.DataFrame): The DataFrame containing the data to be plotted,
                               with columns 'Episode Num' and 'Average Reward'.
        session_number (int): The session number for which the data is being plotted.

    Returns:
        None
    """

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Plot main path
    plt.plot(df['Episode Num'], df['Average Reward'])

    # Add legend, grid, title and labels
    plt.grid(True)
    plt.title(f'{algorithm} Average Reward vs Episode Num for session {session_number}')
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Num')

    # Show the plot
    plt.show()

def generate_traffic_plot(df):

    """
    Visualizes the traffic per charger ID over the timesteps using a graph.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the traffic data to be plotted, with columns 'Charger ID', 'Timestep', and 'Traffic'.

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    # Unique charger ids in your data
    charger_ids = df['Charger ID'].unique()

    # For each charger id, plot a line representing traffic over time
    for charger in charger_ids:
        subset = df[df['Charger ID'] == charger]
        plt.plot(subset['Timestep'], subset['Traffic'], label=str(charger))

    # Add legend, grid, title and labels
    plt.grid(True)
    plt.legend(title="Charger ID")
    plt.title(f'Traffic at Each Station vs Timestep')
    plt.ylabel('Traffic')
    plt.xlabel('Timestep')

    # Show the plot
    plt.show()

def generate_interactive_plot(algorithm, session_number, routes, chargers, origin, destination):

    """
    Visualizes the paths of agents using an interactive graph.

    Parameters:
        algorithm (str): The name of the algorithm used for training.
        session_number (int): The session number for which the data is being plotted.
        routes (list): A list of DataFrames, each containing the route data for an agent, with columns including 'Longitude', 'Latitude', 'Episode Num', 'Agent Num', 'Action', 'Timestep', 'SoC', 'Is Charging', 'Episode Reward', 'Max Time Left', and 'Time to Destination'.
        chargers (pandas.DataFrame): A DataFrame containing the charger data with columns 'Charger ID', 'Latitude', and 'Longitude'.
        origin (list): A list of tuples containing the coordinates of the origin points.
        destination (list): A list of tuples containing the coordinates of the destination points.

    Returns:
        None
    """

    multi_agent = False
    if len(origin) > 1:
       multi_agent = True

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[coord[1] for coord in origin],
        y=[coord[0] for coord in origin],
        mode='markers',
        name='Origin',
        marker=dict(symbol='triangle-up', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[coord[1] for coord in destination],
        y=[coord[0] for coord in destination],
        mode='markers',
        name='Destination',
        marker=dict(symbol='triangle-down', size=10)
    ))

    for idx, route in enumerate(routes, start=1):
        if multi_agent is not True:
            name = f'Path {route.iloc[0][0]}'  # Changed idx to 0 and routes to route
        else:
            name = f'Agent {route.iloc[0][1]}'


        # Plot the path
        fig.add_trace(go.Scatter(
            x=route['Longitude'],
            y=route['Latitude'],
            mode='markers+lines',
            name=name,
            legendgroup=name,
            customdata=route[
                ['Episode Num', 'Agent Num', 'Action', 'Timestep', 'SoC', 'Is Charging', 'Episode Reward', 'Max Time Left', 'Time to Destination']].values.tolist(),
            hovertemplate='Episode: %{customdata[0]}<br>Agent: %{customdata[1]}<br>Time on SOC: %{customdata[7]}<br>Time to Destination: %{customdata[8]}<br>Action: %{customdata[2]}<br>Timestep: %{customdata[3]}<br>SoC: %{customdata[4]}kW<br>Charging: %{customdata[5]}<br>Episode Reward: %{customdata[6]}<br>Lat: %{y}<br>Lon: %{x}'
        ))

        # Plot the last point with a different marker
        fig.add_trace(go.Scatter(
            x=[route['Longitude'].iloc[-1]],
            y=[route['Latitude'].iloc[-1]],
            mode='markers',
            name=f'End {name}',
            marker=dict(symbol='star', size=10),
            legendgroup=name,
            showlegend=False,
            customdata=[
                route[['Episode Num', 'Agent Num', 'Action', 'Timestep', 'SoC', 'Is Charging', 'Episode Reward', 'Max Time Left', 'Time to Destination']].values.tolist()[-1]],
            hovertemplate='Episode: %{customdata[0]}<br>Agent: %{customdata[1]}<br>Time on SOC: %{customdata[7]}<br>Time to Destination: %{customdata[8]}<br>Action: %{customdata[2]}<br>Timestep: %{customdata[3]}<br>SoC: %{customdata[4]}kW<br>Charging: %{customdata[5]}<br>Episode Reward: %{customdata[6]}<br>Lat: %{y}<br>Lon: %{x}'
        ))

    for i in range(len(chargers)):
        fig.add_trace(go.Scatter(
            x=[chargers.iloc[i][2]],
            y=[chargers.iloc[i][1]],
            mode='markers',
            name=f'Charger {int(chargers.iloc[i][0])}'
        ))


    fig.update_layout(
        title=f'{algorithm} Path Visualization for session {session_number}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        annotations=[
            dict(
                x=[coord[1] for coord in origin],
                y=[coord[0] for coord in origin],
                xref="x",
                yref="y",
                text="Origin",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40
            ),
            dict(
                x=[coord[1] for coord in destination],
                y=[coord[0] for coord in destination],
                xref="x",
                yref="y",
                text="Destination",
                showarrow=True,
                arrowhead=2,
                ax=-20,
                ay=40
            )
        ]
    )

    fig.show()

def generate_charger_only_plot(chargers, origin, destination):

    """
    Visualizes the locations of chargers, the origin, and the destination using an interactive graph.

    Parameters:
    chargers (pandas.DataFrame): A DataFrame containing the charger data with columns 'Charger ID', 'Latitude', and 'Longitude'.
    origin (tuple): A tuple containing the coordinates (latitude, longitude) of the origin point.
    destination (tuple): A tuple containing the coordinates (latitude, longitude) of the destination point.

    Returns:
    None
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[origin[0]],
        y=[origin[1]],
        mode='markers',
        name='Origin',
        marker=dict(symbol='triangle-up', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[destination[0]],
        y=[destination[1]],
        mode='markers',
        name='Destination',
        marker=dict(symbol='triangle-down', size=10)
    ))

    for i in range(len(chargers)):
        fig.add_trace(go.Scatter(
            x=[chargers.iloc[i][1]],
            y=[chargers.iloc[i][2]],
            mode='markers',
            name=f'Charger {int(chargers.iloc[i][0])}'
        ))


    fig.update_layout(
        title='Paths',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        annotations=[
            dict(
                x=origin[0],
                y=origin[1],
                xref="x",
                yref="y",
                text="Origin",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40
            ),
            dict(
                x=destination[0],
                y=destination[1],
                xref="x",
                yref="y",
                text="Destination",
                showarrow=True,
                arrowhead=2,
                ax=-20,
                ay=40
            )
        ]
    )

    fig.show()

def visualize_simulation(paths, destinations):

    """
    Visualizes the paths taken by tokens (vehicles) during the simulation along with their destinations.

    Parameters:
        paths (list): A list of token positions at each timestep.
        destinations (torch.Tensor): A tensor containing the coordinates of the destinations.

    Returns:
        None
    """

    # Rearrange the data so that each path represents the movement of one token over time
    token_paths = [torch.array([step[i] for step in paths]) for i in range(len(paths[0]))]

    # Plot the token paths
    colors = ['blue', 'green', 'orange', 'cyan', 'magenta']
    for i, path in enumerate(token_paths):
        plt.plot(path[:, 1], path[:, 0], marker='o', markersize=3, linestyle='-', color=colors[i], label=f'Token {i + 1}')

    # Plot the destinations
    plt.scatter(destinations[:, 1], destinations[:, 0], marker='*', s=100, c='red', label='Destinations')

    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Token Paths and Destinations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def visualize_stats(data, title, measure):

    """
    Visualizes the change in specified measures over time during the simulation.

    Parameters:
        data (list): A list of arrays, where each array contains the values of the specified measure at each timestep.
        title (str): The title of the plot.
        measure (str): The name of the measure being plotted.

    Returns:
        None
    """

    num_steps = len(data)
    num_values = len(data[0])

    # Create a time axis
    time = torch.arange(num_steps)

    # Plot the change in values for each index
    for i in range(num_values):
        values = [array[i] for array in data]
        plt.plot(time, values, label=f'{measure} {i + 1}')

    # Set labels and title
    plt.xlabel('Step')
    plt.ylabel(f'{measure}')
    plt.title(f'{title}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()