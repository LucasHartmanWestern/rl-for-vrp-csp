from .starcraft2.multiagentenv import MultiAgentEnv
from merl_env.environment import EnvironmentClass
import torch
import numpy as np

class Merl_Env_Wrapper(MultiAgentEnv):
    def __init__(self, env, action_dim, chargers, routes):
        self.env = env
        self.observation_space = 2
        self.share_observation_space = env.state_dim - 2
        self.action_space = action_dim
        self.chargers = chargers
        self.routes = routes
        self.num_agents = self.env.num_cars
        self._episode_steps = 0
        self.episode_limit = self.env.max_steps

    def get_available_actions(self):
        return None

    def reset(self):   

        unique_chargers = np.unique(np.array(list(map(tuple, self.chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
        self.env.reset_episode(self.chargers, self.routes, unique_chargers)
        
        local_states = []
        global_state = None
        
        for car in range(self.num_agents): 
            global_state, local_state = self.env.reset_agent(car, False, True)
            local_states.append(local_state)

        available_actions = self.get_available_actions()

        return  local_states, global_state, available_actions

    def step(self, actions):
        # Initialize and perform the environment steps for all agents
        global_state = None
        local_states = []
        self.env.init_routing()
        for car in range(self.num_agents):
            global_state, local_state = self.env.reset_agent(car, False, True)
            local_states.append(local_state)
            self.env.generate_paths(actions[car], None, car)
        
        self._episode_steps += 1
        sim_done, arrived_at_final = self.env.simulate_routes(self._episode_steps)
        sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = self.env.get_results()
        
        # Expand the dimensions of time_step_rewards to make it compatible with rtgs
        time_step_rewards = np.expand_dims(time_step_rewards, axis=-1)  # Now it should have the shape (env.n_threads, env.num_agents, 1)
    
        # Convert dones to a numpy array (if not already)
        dones = arrived_at_final.detach().cpu().numpy()
        
        return local_states, global_state, time_step_rewards, dones, None, None

    def close(self):
        # Check if the environment exists and has a close method
        if self.env and hasattr(self.env, 'close'):
            try:
                self.env.close()  # Attempt to close the environment if possible
            except Exception as e:
                print(f"Error while closing the environment: {e}")
    
        # If you're using any specific resources such as PyTorch tensors, make sure to clean them up
        # e.g., if there are any PyTorch tensors, call .detach() or .cpu() if necessary
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
    
        # Any other cleanup tasks, such as resetting variables or freeing memory
        self.env = None
        self.chargers = None
        self.routes = None
        print("Environment has been successfully closed.")

        