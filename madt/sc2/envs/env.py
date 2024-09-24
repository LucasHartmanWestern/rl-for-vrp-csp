from .starcraft2.StarCraft2_Env import StarCraft2Env
from .starcraft2.smac_maps import get_map_params
from .config import get_config
from .env_wrappers import ShareSubprocVecEnv
from .Merl_Env_Wrapper import Merl_Env_Wrapper



def make_eval_env(all_args, merl_env, action_dim, chargers, routes, n_threads=1):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
                env.seed(all_args.seed * 50000 + rank * 10000)
            elif all_args.env_name == "Merl":
                env = Merl_Env_Wrapper(merl_env, action_dim, chargers, routes)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            
            return env

        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


class Env:
    def __init__(self, merl_env, action_dim, chargers, routes, n_threads=1):
        parser = get_config()
        all_args = parser.parse_known_args()[0]
        self.real_env = make_eval_env(all_args, merl_env, action_dim, chargers, routes, n_threads)
        self.num_agents = 3
        self.max_timestep = 7
        self.n_threads = n_threads