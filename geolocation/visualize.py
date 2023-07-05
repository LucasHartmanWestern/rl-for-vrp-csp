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


def generate_average_reward_plot(algorithm, df):
    """Visualize the data using a graph.
        The graph shows the average reward over time per episode"""

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Plot main path
    plt.plot(df['Episode Num'], df['Average Reward'])

    # Add legend, grid, title and labels
    plt.grid(True)
    plt.title(f'{algorithm} Average Reward vs Episode Num')
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Num')

    # Show the plot
    plt.show()

def generate_interactive_plot(algorithm, routes, chargers, origin, destination):
    """Visualize the data using an interactive graph."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[origin[1]],
        y=[origin[0]],
        mode='markers',
        name='Origin',
        marker=dict(symbol='triangle-up', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[destination[1]],
        y=[destination[0]],
        mode='markers',
        name='Destination',
        marker=dict(symbol='triangle-down', size=10)
    ))

    for idx, route in enumerate(routes, start=1):
        name = f'Path {route.iloc[0][0]}'  # Changed idx to 0 and routes to route

        # Plot the path
        fig.add_trace(go.Scatter(
            x=route['Longitude'],
            y=route['Latitude'],
            mode='markers+lines',
            name=name,
            legendgroup=name,
            customdata=route[
                ['Episode Num', 'Action', 'Timestep', 'SoC', 'Is Charging', 'Episode Reward']].values.tolist(),
            hovertemplate='Episode: %{customdata[0]}<br>Action: %{customdata[1]}<br>Timestep: %{customdata[2]}<br>SoC: %{customdata[3]}kW<br>Charging: %{customdata[4]}<br>Episode Reward: %{customdata[5]}<br>Lat: %{y}<br>Lon: %{x}'
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
                route[['Episode Num', 'Action', 'Timestep', 'SoC', 'Is Charging', 'Episode Reward']].values.tolist()[-1]],
            hovertemplate='Episode: %{customdata[0]}<br>Action: %{customdata[1]}<br>Timestep: %{customdata[2]}<br>SoC: %{customdata[3]}kW<br>Charging: %{customdata[4]}<br>Episode Reward: %{customdata[5]}<br>Lat: %{y}<br>Lon: %{x}'
        ))

    for i in range(len(chargers)):
        fig.add_trace(go.Scatter(
            x=[chargers.iloc[i][2]],
            y=[chargers.iloc[i][1]],
            mode='markers',
            name=f'Charger {int(chargers.iloc[i][0])}'
        ))


    fig.update_layout(
        title=f'{algorithm} Path Visualization',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        annotations=[
            dict(
                x=origin[1],
                y=origin[0],
                xref="x",
                yref="y",
                text="Origin",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40
            ),
            dict(
                x=destination[1],
                y=destination[0],
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