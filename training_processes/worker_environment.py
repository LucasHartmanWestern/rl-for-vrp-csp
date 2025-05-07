import sys
import os
import time
import torch
import numpy as np
from merl_env.environment import EnvironmentClass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_config_file
class worker_environment:
    '''Worker which will stand as an MP subprocess to train agents
    for one episode in the environment. Data will be exchanged using
    a communication channel.
    '''

    def __init__(self, experiment_number, zone, rank, charger_seed, device, dtype=torch.float32):
        '''Initiates the worker and the environment
        
        '''
        self.running = True
        self.rank = rank
        self.worker_zone = zone
        self.device = device
        self.dtype = dtype

        #Load environment configuration
        config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
        c = load_config_file(config_fname)
       
        seed = c['environment_settings']['seed']
        coords = c['environment_settings']['coords'][zone]
        self.environment = EnvironmentClass(config_fname, seed, charger_seed, coords,\
                                            device=device, dtype=dtype)
        self.num_cars = c['environment_settings']['num_of_cars']
        self.agent_by_zone = c['algorithm_settings']['agent_by_zone']
        self.num_agents = 1 if self.agent_by_zone else self.num_cars
        self.cumulative_flag = c['nn_hyperparameters']['average_rewards_when_training']
        self.max_generations = c['cma_parameters']['max_generations']
        self.current_gen = 0
        

        print(f'Zone {zone} worker {rank} started.')


    def run_episode(self, agents_by_pop):
        # # Reset just this candidate’s episode
        self.environment.reset_episode(self.chargers, self.routes, self.unique_chargers)

        self.episode_metrics = []
        self.episode_rewards = []
        # environment.cma_copy_store()  # Restore environment to its stored state
        sim_done = False
        timestep = 0
        # cumulative reward: scalar if agent_by_zone else vector per car
        rewards_cumulative = torch.zeros(self.num_agents, device=self.device)

        while not sim_done:
            self.environment.init_routing()
            start_time_step = time.time()
            
            for car_idx in range(self.num_cars):
                state = self.environment.reset_agent(car_idx, timestep)
                agent_idx = 0 if self.agent_by_zone else car_idx
                car_route = agents_by_pop[agent_idx]['structure'](state)
                self.environment.generate_paths(car_route, None, agent_idx)

            sim_done = self.environment.simulate_routes(timestep)
            sim_path_results, sim_traffic, sim_battery_levels, sim_distances,\
                timestep_rewards, arrived_at_final = self.environment.get_results()

            # self.episode_rewards[timestep,:] = timestep_rewards
            if self.agent_by_zone:
                # one agent sees the mean across all cars
                rewards_cumulative[0] += timestep_rewards.mean()
            elif self.cumulative_flag: 
                # rewards_cumulative +=  torch.full((self.num_cars,),\
                #                           timestep_rewards.sum(axis=1).mean(),\
                #                           device=self.device)
                rewards_cumulative +=  torch.full((self.num_cars,),\
                                          timestep_rewards.mean(),\
                                          device=self.device)
            else:
                # each agent gets its own car’s reward
                rewards_cumulative += timestep_rewards

            self.episode_metrics.append({
                "zone": self.worker_zone,
                "episode": self.current_gen,
                "timestep": timestep,
                "aggregation": self.aggregation_num,
                "paths": sim_path_results,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": timestep_rewards,
                "best_reward": self.best_avg,
                "timestep_real_world_time": time.time() - start_time_step,
                "done": sim_done
            })
            
            timestep += 1
        
        self.current_gen +=1
        
        return rewards_cumulative

    def init_environment(self, routes, chargers, aggregation_num):
        self.routes = routes
        self.chargers = chargers
        self.aggregation_num = aggregation_num
        self.unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))
        
        self.current_gen = 0
        # self.episode_rewards = torch.zeros((self.num_cars), device=self.device)
        self.best_avg = float('-inf')

    def get_ev_info(self):
        return self.environment.get_ev_info()
        

    def stop(self):
        self.environment = None
        self.episode_metrics = None
        self.running = False


    def run(self, pop_idx, comm):
        while self.running:
            if comm.poll():  # Check if parent sent a message
                action = comm.recv()
    
                if action == "init_environment":
                    routes, chargers, aggregation_num = comm.recv()
                    self.init_environment(routes, chargers, aggregation_num)
                
                elif action == "run_episode":
                    agents_by_pop = comm.recv()
                    cumulative = self.run_episode(agents_by_pop)
                    #Sending back reward episode results to parent
                    comm.send((self.rank, cumulative))

                elif action == "get_metrics":
                    comm.send(self.episode_metrics)

                elif action == "best_avg":
                    self.best_avg = comm.recv()

                elif action == "stop":
                    self.stop()
                    break
        
                else:
                    print(f"{self.rank} in zone {self.worker_zone} Unknown command: {action}")
            else:
                time.sleep(0.001)  # idle wait
        #closing worker
        comm.close()
        print(f"[Zone {self.worker_zone} Worker {self.rank}] stopping.")



        