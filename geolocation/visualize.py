import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_excel_data(file_path, sheet_name):
    """Retrieve the data from the excel file and parse it as needed"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_plot(df):
    """Visualize the data using a graph.
    The graph connects the coordinates in the Latitude and Longitude columns
    and plots the coordinates in the various (CX_Latitude, CX_Longitude) columns."""

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

def plot_aggregate_reward_data(data):
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
    """Visualize the data using a graph.
        The graph shows the average reward over time per episode"""

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
    """Visualize the traffic per charger id over the timesteps."""

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
    """Visualize the data using an interactive graph."""

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
    """Visualize the data using an interactive graph."""

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