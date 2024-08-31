from starcraft2.multiagentenv import MultiAgentEnv
from merl_env.environment import EnvironmentClass
import torch


class Merl_Env_Wrapper(MultiAgentEnv):
    def __init__(self, env, action_dim):
        self.env = env
        self.observation_space = 2
        self.share_observation_space = env.state_dim - 2
        self.action_space = action_dim
        self.num_agents = self.env.num_cars
        self._episode_steps = 0
        self.episode_limit = self.env.max_steps

    def get_available_actions(self):
        return None

    def reset(self):
        
        self.env.reset_episode(chargers, routes, unique_chargers)
        
        local_states = []
        global_state = None
        
        for car in range(self.num_agents): 
            global_state, local_state = self.env.reset_agent(car, False, True)
            local_states.append(local_state)

        available_actions = self.get_available_actions()

        return global_state, local_states, None

    def step(self, actions):
        global_state = None
        local_states = []
        self.env.init_routing()
        for car in range (num_cars):
            global_state, local_state = self.env.reset_agent(car, False, True)
            local_states.append(local_state)
            self.env.generate_paths(actions[car], None, car)
            
        self._episode_steps += 1
        sim_done, arrived_at_final = self.env.simulate_routes(self._episode_steps)
        sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = self.env.get_results()

        return local_states, global_state, time_step_rewards, arrived_at_final, None
        