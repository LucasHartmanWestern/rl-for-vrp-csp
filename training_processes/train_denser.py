# Adapted by Santiago August 9, 2024
import os
import time
import copy
import torch
import numpy as np

from data_loader import load_config_file
from agents.denser_agent import DenserAgent
from evaluation import evaluate
from merl_env._pathfinding import haversine

def train_denser(ev_info, 
                 metrics_base_path,
                 experiment_number,
                 chargers, environment,
                 routes, date,
                 action_dim,
                 global_weights,
                 aggregation_num,
                 zone_index,
                 seed,
                 main_seed,
                 device,
                 agent_by_zone,
                 variant,
                 args,
                 fixed_attributes,
                 verbose,
                 display_training_times=False,
                 dtype=torch.float32,
                 save_offline_data=False,
                 train_model=True,
                 old_buffers=None):
    """
    Trains decision-making agents using the DENSER algorithm with a grammar-based representation.
    This updated version calls the agents’ decoded PyTorch models (structure) to get outputs.
    """
    start_time = time.time()
    avg_rewards = []
    seed = int(seed)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    unique_chargers = np.unique(
        np.array(list(map(tuple, chargers.reshape(-1, 3))),
                 dtype=[('id', int), ('lat', float), ('lon', float)])
    )
    state_dimension = (environment.num_chargers * 3 * 2) + 6
    model_indices = environment.info['model_indices']

    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eps_per_save = int(nn_c['eps_per_save'])
    num_episodes = nn_c['num_episodes'] if not args.eval else 100

    num_cars = environment.num_cars
    num_agents = 1 if agent_by_zone else num_cars

    denser_agents_list = []
    for agent_idx in range(num_agents):
        initial_weights = None
        if global_weights is not None:
            if agent_idx < len(global_weights):
                initial_weights = global_weights[agent_idx]
            else:
                print(f"Warning: No global weights found for agent {agent_idx}")
        agent = DenserAgent(state_dimension, action_dim, num_cars, seed, agent_idx, initial_weights, experiment_number, device)
        denser_agents_list.append(agent)

    avg_output_values = torch.zeros((denser_agents_list[0].max_generation, action_dim), device=device)
    best_avg = float('-inf')
    metrics = []

    denser_info = denser_agents_list[0]
    population_size = denser_info.population_size

    # try:
    #     environment.cma_store()
    # except:
    #     print("Error storing environment")
    # Reset the environment for a new training episode
    environment.reset_episode(chargers, routes, unique_chargers)
    environment.cma_store()
    fitnesses = torch.full((population_size, num_agents), float('-inf'), device=device)

    for generation in range(denser_info.max_generation):
        # environment.reset_episode(chargers, routes, unique_chargers)
        # fitnesses = torch.zeros((population_size, num_agents), device=device)

        # --- Evaluate each candidate in the current population, one full episode at a time ---
        for pop_idx in range(population_size):
            # # Reset just this candidate’s episode
            # environment.reset_episode(chargers, routes, unique_chargers)

            environment.cma_copy_store()  # Restore environment to its stored state
            sim_done = False
            timestep = 0
            # cumulative reward: scalar if agent_by_zone else vector per car
            cumulative = torch.zeros(num_agents, device=device)

            while not sim_done:
                environment.init_routing()

                for car_idx in range(num_cars):
                    state = environment.reset_agent(car_idx, timestep)
                    agent_idx = 0 if agent_by_zone else car_idx
                    agent = denser_agents_list[agent_idx]
                    candidate = agent.population[pop_idx]
                    state_tensor = torch.tensor(state, dtype=dtype, device=device)
                    car_route = candidate['structure'](state_tensor)
                    environment.generate_paths(car_route, None, agent_idx)

                sim_done = environment.simulate_routes(timestep)
                _, _, _, _, rewards_pop, _ = environment.get_results()

                # print(f'pop {pop_idx}, timestep {timestep} rewards pop size {rewards_pop.shape} matrix{rewards_pop}')
                # print(f'reward cumulative {rewards_pop.sum(axis=0).mean()}')
                if agent_by_zone:
                    # one agent sees the mean across all cars
                    cumulative[0] += rewards_pop.sum(axis=0).mean()
                elif 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
                    cumulative +=  torch.full((num_cars,),rewards_pop.sum(axis=0).mean(), device=device)
                else:
                    # each agent gets its own car’s reward
                    cumulative += rewards_pop

                timestep += 1

            # assign the total fitness for this individual
            if agent_by_zone:
                fitnesses[pop_idx, 0] = cumulative[0]
            else:
                fitnesses[pop_idx, :] = cumulative

        # Update each agent with the fitness values
        for agent_idx, agent in enumerate(denser_agents_list):
            agent.tell(fitnesses[:, agent_idx].flatten())

        # --- Evaluate the best individual per agent for logging & metrics ---
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False
        timestep_counter = 0
        rewards = []

        while not sim_done:
            environment.init_routing()
            start_time_step = time.time()

            for car_idx in range(num_cars):
                state = environment.reset_agent(car_idx, timestep_counter)
                agent_idx = 0 if agent_by_zone else car_idx
                agent = denser_agents_list[agent_idx]
                best_model = agent.best_individual['structure']
                state_tensor = torch.tensor(state, dtype=dtype, device=device)
                car_route = best_model(state_tensor)
                environment.generate_paths(car_route, None, agent_idx)

            sim_done = environment.simulate_routes(timestep_counter)
            sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards, arrived_at_final = environment.get_results()

            if timestep_counter == 0:
                episode_rewards = np.expand_dims(time_step_rewards, axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards, time_step_rewards))

            if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']:
                avg_r = time_step_rewards.sum(axis=0).mean()
                rewards.extend([avg_r] * len(time_step_rewards))
            else:
                rewards.extend(time_step_rewards)

            metrics.append({
                "zone": zone_index,
                "episode": generation,
                "timestep": timestep_counter,
                "aggregation": aggregation_num,
                "paths": sim_path_results,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": time_step_rewards,
                "best_reward": best_avg,
                "timestep_real_world_time": time.time() - start_time_step,
                "done": sim_done
            })
            timestep_counter += 1

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed))

        if verbose:
            elapsed_time = time.time() - start_time
            print_log(
                f'(Aggregation: {aggregation_num+1} Zone: {zone_index+1} '
                f'Generation: {generation+1}/{denser_info.max_generation}) - '
                f'avg reward {avg_reward:.3f}',
                date, elapsed_time
            )

        # periodic save
        if ((generation + 1) % eps_per_save == 0 and generation > 0 and train_model) \
           or (generation == denser_info.max_generation - 2):
            metrics_path = f"{metrics_base_path}/{'eval' if args.eval else 'train'}"
            os.makedirs(metrics_path, exist_ok=True)
            evaluate(ev_info, metrics, seed, date, verbose, 'save',
                     num_episodes, f"{metrics_path}/metrics", True)
            metrics = []

        if avg_reward > best_avg:
            best_avg = avg_reward
            if verbose:
                print_log(
                    f'Zone: {zone_index+1} Gen: {generation+1}/{denser_info.max_generation} '
                    f'- New Best: {best_avg:.3f}',
                    date, None
                )

        # log dummy outputs
        dummy_state = torch.zeros(state_dimension, dtype=dtype, device=device)
        gen_outputs = torch.empty((len(denser_agents_list),
                                   len(denser_agents_list[0].best_individual['structure'](dummy_state))),device=device)
        for idx, agent in enumerate(denser_agents_list):
            gen_outputs[idx] = agent.best_individual['structure'](dummy_state).detach()
        # gen_outputs = [
        #     agent.best_individual['structure'](dummy_state)
        #     .detach().cpu().numpy()
        #     for agent in denser_agents_list
        # ]
        # avg_output_values[generation] = np.mean(gen_outputs, axis=0)
        avg_output_values[generation] = gen_outputs.mean(axis=0)

    # End of evolution
    sim_path_results, sim_traffic, sim_battery_levels, sim_distances, rewards, arrived_at_final = \
        environment.get_results()
    print(f'Rewards for population evolution: {np.mean(rewards):.3f} '
          f'after {denser_info.max_generation} generations')

    # save networks
    folder_path = 'saved_networks'
    os.makedirs(folder_path, exist_ok=True)
    fname = f'{folder_path}/denser_model_{main_seed}_z{zone_index}'
    for idx, agent in enumerate(denser_agents_list):
        agent.save_model(f'{fname}_agent{idx}.pkl')

    elapsed_time = time.time() - start_time
    # weights_list = [agent.get_weights() for agent in denser_agents_list]
    structure_list = [agent.get_best_solutions() for agent in denser_agents_list]

    return structure_list, avg_rewards, avg_output_values.cpu(), metrics, None


def print_log(label, date, et):
    if et is not None:
        to_print = f"{label}\t - et " \
                   f"{int(et//3600):02}:{int((et//60)%60):02}:" \
                   f"{int(et%60):02}.{int((et*1000)%1000)}"
    else:
        to_print = label
    log_fname = f'logs/{date}-training_logs.txt'
    os.makedirs(os.path.dirname(log_fname), exist_ok=True)
    with open(log_fname, 'a') as file:
        print(to_print, file=file)
    print(to_print)
