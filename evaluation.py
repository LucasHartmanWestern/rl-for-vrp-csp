import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
from data_loader import save_to_json, load_from_json
import mplcursors

def evaluate(ev_info, metrics, seed, date, verbose, purpose, num_episodes, base_path):
    if purpose == 'save':

        traffic_data = []
        distance_data = []
        reward_data = []
        battery_data = []
        time_data = []
        path_data = []

        # Get the model index by using car_models[zone_index][agent_index]
        car_models = np.column_stack([info['model_type'] for info in ev_info]).T

        st = time.time()

        # Flatten the data
        for zone_agg in metrics:
            for episode in zone_agg:

                # Loop through sim steps and stations
                for step_ind in range(len(episode['traffic'])):
                    for station_ind in range(len(episode['traffic'][0])):
                        traffic_data.append({
                            "episode": episode['episode'],
                            "timestep": episode['timestep'],
                            "done": episode['done'],
                            "zone": episode['zone'] + 1,
                            "aggregation": episode['aggregation'],
                            "simulation_step": step_ind,
                            "station_index": station_ind,
                            "traffic": episode['traffic'][step_ind][station_ind]
                        })

                # Loop through the agents in each zone
                for agent_ind, car_model in enumerate(car_models[episode['zone']]):

                    distance_data.append({
                        "episode": episode['episode'],
                        "timestep": episode['timestep'],
                        "done": episode['done'],
                        "zone": episode['zone'] + 1,
                        "aggregation": episode['aggregation'],
                        "agent_index": agent_ind,
                        "car_model": car_model,
                        "distance": episode['distances'][-1][agent_ind] * 100
                    })

                    reward_data.append({
                        "episode": episode['episode'],
                        "timestep": episode['timestep'],
                        "done": episode['done'],
                        "zone": episode['zone'] + 1,
                        "aggregation": episode['aggregation'],
                        "agent_index": agent_ind,
                        "car_model": car_model,
                        "reward": episode['rewards'][agent_ind]
                    })

                    steps_taken = np.array(episode['distances']).T[agent_ind]

                    time_data.append({
                        "episode": episode['episode'],
                        "timestep": episode['timestep'],
                        "done": episode['done'],
                        "zone": episode['zone'] + 1,
                        "aggregation": episode['aggregation'],
                        "agent_index": agent_ind,
                        "car_model": car_model,
                        "duration": np.where(steps_taken == steps_taken[-1])[0][0]
                    })

                    agent_battery = np.array(episode['batteries']).T[agent_ind]

                    battery_data.append({
                        "episode": episode['episode'],
                        "timestep": episode['timestep'],
                        "done": episode['done'],
                        "zone": episode['zone'] + 1,
                        "aggregation": episode['aggregation'],
                        "agent_index": agent_ind,
                        "car_model": car_model,
                        "average_battery": np.average(agent_battery),
                        "ending_battery": agent_battery.tolist()[-1],
                        "starting_battery": agent_battery.tolist()[0]
                    })

                    path_data.append({
                        "episode": episode['episode'],
                        "timestep": episode['timestep'],
                        "done": episode['done'],
                        "zone": episode['zone'] + 1,
                        "aggregation": episode['aggregation'],
                        "agent_index": agent_ind,
                        "origin": episode['paths'][0][agent_ind].tolist(),
                        "destination": episode['paths'][-1][agent_ind].tolist(),
                        "path": [step.tolist() for step in episode['paths'][:, agent_ind]]
                    })

        et = time.time() - st

        if verbose:
            print(f'\nSpent {et:.3f} seconds reformatting the results for evaluation\n')  # Print saving time with 3 decimal places

        st = time.time()

        save_to_json(distance_data, f'{base_path}_distance.json')
        save_to_json(battery_data, f'{base_path}_battery.json')
        save_to_json(time_data, f'{base_path}_time.json')
        save_to_json(reward_data, f'{base_path}_reward.json')
        save_to_json(traffic_data, f'{base_path}_traffic.json')
        save_to_json(path_data, f'{base_path}_path.json')

        et = time.time() - st

        if verbose: print(f'\nSpent {et:.3f} seconds saving the results for evaluation\n')  # Print saving time with 3 decimal places

    if purpose == 'display':

        distance_data = load_from_json(f'{base_path}_distance.json')
        battery_data = load_from_json(f'{base_path}_battery.json')
        time_data = load_from_json(f'{base_path}_time.json')
        reward_data = load_from_json(f'{base_path}_reward.json')
        traffic_data = load_from_json(f'{base_path}_traffic.json')
        path_data = load_from_json(f'{base_path}_path.json')

        # Draw a map of the last episode
        draw_map_of_last_episode(path_data, seed)

        # Evaluate the metrics per-agent
        evaluate_by_agent(distance_data, 'distance', 'Distance Travelled (km)', seed, verbose, num_episodes)
        evaluate_by_agent(battery_data, 'average_battery', 'Battery Level (Watts)', seed, verbose, num_episodes)
        evaluate_by_agent(battery_data, 'ending_battery', 'Ending Battery Level (Watts)', seed, verbose, num_episodes)
        evaluate_by_agent(time_data, 'duration', 'Time Spent Travelling (Steps)', seed, verbose, num_episodes)
        evaluate_by_agent(reward_data, 'reward', 'Simulation Reward', seed, verbose, num_episodes)

        # Evaluate metrics per-station
        evaluate_by_station(traffic_data, seed, verbose, num_episodes)

def evaluate_by_agent(data, metric_name, metric_title, seed, verbose, num_episodes, algorithm='DQN'):
    if verbose: print(f"Evaluating {metric_title} Metrics for seed {seed}")

    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Filter data to only include the last timestep within each episode
    df = df[df['done'] == True]

    # Get recalculated episodes using (aggregation number * episodes per aggregation) + episode number
    df['recalculated_episode'] = df['aggregation'] * num_episodes + df['episode']

    # Evaluate average battery level throughout simulation
    avg_total = df[metric_name].mean()

    # Evaluate average throughout simulation by zone
    avg_by_zone = df.groupby('zone')[metric_name].mean()

    # Evaluate average throughout simulation by car model
    avg_by_car_model = df.groupby('car_model')[metric_name].mean()

    # Evaluate average throughout simulation by aggregation
    avg_by_aggregation = df.groupby('aggregation')[metric_name].mean()

    # Evaluate average per episode of training
    avg_by_episode = df.groupby('recalculated_episode')[metric_name].mean()

    # Evaluate average per episode of training by zone
    avg_by_episode_zone = df.groupby(['recalculated_episode', 'zone'])[metric_name].mean().unstack()

    # Evaluate average per episode of training by car model
    avg_by_episode_car_model = df.groupby(['recalculated_episode', 'car_model'])[metric_name].mean().unstack()

    # Evaluate average per episode of training by aggregation
    avg_by_episode_aggregation = df.groupby(['episode', 'aggregation'])[metric_name].mean().unstack()

    # Average Total
    plt.figure(figsize=(8, 6))
    plt.bar(['Average'], [avg_total])
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title}')
    plt.show()

    # Average by Zone
    plt.figure(figsize=(8, 6))
    plt.bar(avg_by_zone.index, avg_by_zone.values)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} by Zone')
    plt.show()

    # Average by Car Model
    plt.figure(figsize=(8, 6))
    plt.bar(avg_by_car_model.index, avg_by_car_model.values)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} by Car Model')
    plt.show()

    # Average by Aggregation
    plt.figure(figsize=(8, 6))
    plt.bar(avg_by_aggregation.index, avg_by_aggregation.values)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} by Aggregation')
    plt.show()

    # Average per Episode
    plt.figure(figsize=(8, 6))
    plt.plot(avg_by_episode.index, avg_by_episode.values)
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} per Episode')
    plt.show()

    # Average per Episode by Zone
    plt.figure(figsize=(8, 6))
    avg_by_episode_zone.plot()
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} per Episode by Zone')
    plt.show()

    # Average per Episode by Car Model
    plt.figure(figsize=(8, 6))
    avg_by_episode_car_model.plot()
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - A verage {metric_title} per Episode by Car Model')
    plt.show()

    # Average per Episode by Aggregation
    plt.figure(figsize=(8, 6))
    avg_by_episode_aggregation.plot()
    plt.ylabel(f'{metric_title}')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average {metric_title} per Episode by Aggregation')
    plt.show()

def draw_map_of_last_episode(data, seed, algorithm='DQN'):
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Get the last episode
    last_episode = df['episode'].max()

    # Get the data for the last episode
    last_episode_data = df[df['episode'] == last_episode]

    # Get unique zones and aggregations
    unique_zones = last_episode_data['zone'].unique()
    unique_aggregations = last_episode_data['aggregation'].unique()

    # Plot each zone on a different graph
    for zone in unique_zones:
        zone_data = last_episode_data[last_episode_data['zone'] == zone]

        plt.figure(figsize=(20, 16))  # Make the plot larger
        plt.title(f'Seed {seed} - Algo. {algorithm} - Zone {zone} - Last Episode Paths')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        paths = []

        # Plot each aggregation on a different graph
        for aggregation in unique_aggregations:
            agg_data = zone_data[zone_data['aggregation'] == aggregation]

            for agent_index in agg_data['agent_index'].unique():
                agent_data = agg_data[agg_data['agent_index'] == agent_index]

                # Extract combined path, origin, and destination
                combined_path = np.vstack(agent_data['path'].values)
                origin = combined_path[0]
                destination = combined_path[-1]

                # Plot path with smaller dots
                path_line, = plt.plot(combined_path[:, 0], combined_path[:, 1], marker='o', markersize=3, label=f'Agent {agent_index} Path')
                paths.append(path_line)

                # Plot origin and destination
                plt.scatter(origin[0], origin[1], marker='^', s=100, label=f'Agent {agent_index} Origin')
                plt.scatter(destination[0], destination[1], marker='*', s=100, label=f'Agent {agent_index} Destination')

        # Add interactive cursor for paths
        cursor = mplcursors.cursor(paths, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            for path in paths:
                path.set_alpha(0.1)  # Make all paths semi-transparent
            sel.artist.set_alpha(1.0)  # Highlight the selected path
            plt.draw()

        plt.show()

    # Plot all zones together
    plt.figure(figsize=(20, 16))  # Make the plot larger
    plt.title(f'Seed {seed} - Algo. {algorithm} - All Zones - Last Episode Paths')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    paths = []

    for zone in unique_zones:
        zone_data = last_episode_data[last_episode_data['zone'] == zone]

        for aggregation in unique_aggregations:
            agg_data = zone_data[zone_data['aggregation'] == aggregation]

            for agent_index in agg_data['agent_index'].unique():
                agent_data = agg_data[agg_data['agent_index'] == agent_index]

                # Extract combined path, origin, and destination
                combined_path = np.vstack(agent_data['path'].values)
                origin = combined_path[0]
                destination = combined_path[-1]

                # Plot path with smaller dots
                path_line, = plt.plot(combined_path[:, 0], combined_path[:, 1], marker='o', markersize=3, label=f'Zone {zone} Agent {agent_index} Path')
                paths.append(path_line)

                # Plot origin and destination
                plt.scatter(origin[0], origin[1], marker='^', s=100, label=f'Zone {zone} Agent {agent_index} Origin')
                plt.scatter(destination[0], destination[1], marker='*', s=100, label=f'Zone {zone} Agent {agent_index} Destination')

    # Add interactive cursor for paths
    cursor = mplcursors.cursor(paths, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        for path in paths:
            path.set_alpha(0.1)  # Make all paths semi-transparent
        sel.artist.set_alpha(1.0)  # Highlight the selected path
        plt.draw()

    plt.show()

def evaluate_training_duration(data, algorithm='DQN'):
    print("Evaluating Training Time Metrics")

    # TODO:
    # - Evaluate how long it takes to plateau to reward
    # - Evaluate how long it takes to retrain after defining base models

def evaluate_by_station(data, seed, verbose, num_episodes, algorithm='DQN'):
    if verbose: print("Evaluating Traffic Metrics")

    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Filter data to only include the last timestep within each episode
    df = df[df['done'] == True]

    # Get recalculated episodes using (aggregation number * episodes per aggregation) + episode number
    df['recalculated_episode'] = df['aggregation'] * num_episodes + df['episode']

    # Add peak traffic by charger (station_index)
    peak_traffic_by_charger = df.groupby('station_index')['traffic'].max()

    # Add average traffic levels
    average_traffic_levels = df['traffic'].mean()

    # Add traffic levels across zones
    traffic_levels_across_zones = df.groupby('zone')['traffic'].mean()

    # Add traffic levels across aggregations
    traffic_levels_across_aggregations = df.groupby('aggregation')['traffic'].mean()

    # Add average traffic per episode of training
    average_traffic_per_episode = df.groupby('recalculated_episode')['traffic'].mean()

    # Add average traffic per episode of training by zone
    average_traffic_per_episode_by_zone = df.groupby(['recalculated_episode', 'zone'])['traffic'].mean()

    # Add average traffic per episode of training by aggregation
    average_traffic_per_episode_by_aggregation = df.groupby(['episode', 'aggregation'])['traffic'].mean()

    # Add peak traffic for each charger in the last episode
    last_episode = df['episode'].max()
    peak_traffic_last_episode = df[df['episode'] == last_episode].groupby('station_index')['traffic'].max()

    # Add average peak traffic per episode
    peak_traffic_per_episode = df.groupby(['recalculated_episode', 'station_index'])['traffic'].max()
    average_peak_traffic_per_episode = peak_traffic_per_episode.groupby('recalculated_episode').mean()

    # Peak Traffic by Charger
    plt.figure(figsize=(8, 6))
    peak_traffic_by_charger.plot(kind='bar', color='skyblue')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Peak Traffic by Charger Throughout Training')
    plt.xlabel('Station Index')
    plt.ylabel('Peak Traffic')
    plt.show()

    # Peak Traffic by Charger in the Last Episode
    plt.figure(figsize=(8, 6))
    peak_traffic_last_episode.plot(kind='bar', color='purple')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Peak Traffic by Charger in Last Episode')
    plt.xlabel('Station Index')
    plt.ylabel('Peak Traffic')
    plt.show()

    # Average Traffic Levels
    plt.figure(figsize=(8, 6))
    plt.bar(['Average Traffic'], [average_traffic_levels], color='lightgreen')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Levels')
    plt.ylabel('Average Traffic')
    plt.show()

    # Traffic Levels Across Zones
    plt.figure(figsize=(8, 6))
    traffic_levels_across_zones.plot(kind='bar', color='salmon')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Levels Across Zones')
    plt.xlabel('Zone')
    plt.ylabel('Average Traffic')
    plt.show()

    # Traffic Levels Across Aggregations
    plt.figure(figsize=(8, 6))
    traffic_levels_across_aggregations.plot(kind='bar', color='orange')
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Levels Across Aggregations')
    plt.xlabel('Aggregation')
    plt.ylabel('Average Traffic')
    plt.show()

    # Average Traffic Per Episode of Training
    plt.figure(figsize=(8, 6))
    average_traffic_per_episode.plot()
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Per Episode of Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Traffic')
    plt.show()

    # Average Peak Traffic Per Episode
    plt.figure(figsize=(8, 6))
    average_peak_traffic_per_episode.plot()
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Peak Traffic Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Peak Traffic')
    plt.show()

    # Average Traffic Per Episode of Training by Zone
    plt.figure(figsize=(8, 6))
    average_traffic_per_episode_by_zone.unstack().plot()
    for x in range(0, max(df['recalculated_episode']) + 2, num_episodes):
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.75)
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Per Episode by Zone')
    plt.xlabel('Episode')
    plt.ylabel('Average Traffic')
    plt.legend(title='Zone')
    plt.show()

    # Average Traffic Per Episode of Training by Aggregation
    plt.figure(figsize=(8, 6))
    average_traffic_per_episode_by_aggregation.unstack().plot()
    plt.title(f'Seed {seed} - Algo. {algorithm} - Average Traffic Per Episode by Aggregation')
    plt.xlabel('Episode')
    plt.ylabel('Average Traffic')
    plt.legend(title='Aggregation')
    plt.show()

if __name__ == '__main__': # For debugging

    from data_loader import load_config_file
    from datetime import datetime

    environment_config_fname = 'configs/environment_config.yaml'
    nn_config_fname = 'configs/neural_network_config.yaml'

    c = load_config_file(environment_config_fname)
    env_c = c['environment_settings']
    c = load_config_file(nn_config_fname)
    nn_c = c['nn_hyperparameters']

    seed = env_c['seeds'][0]
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    num_episodes = nn_c['num_episodes']
    attr_label = 'learned'

    evaluate(None, None, seed, date, True, 'display', num_episodes, f"metrics/train/metrics_{env_c['num_of_cars']}_{num_episodes}_{seed}_{attr_label}")
