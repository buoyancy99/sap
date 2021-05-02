from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments.gridworld_env import Grid_World_Env


class Grid_World_Vec_Env(SubprocVecEnv):
    def __init__(self, num_envs, reward_map, reward_map_hat, seed =0):

        def make_grid_world_env(rank):
            def _thunk():
                robot_env = Grid_World_Env(reward_map, reward_map_hat)
                return robot_env

            return _thunk

        self.num_envs = num_envs
        super().__init__([make_grid_world_env(i) for i in range(num_envs)])


    def backup(self):
        self.env_method('backup', indices=range(self.num_envs))

    def restore(self):
        self.env_method('restore', indices=range(self.num_envs))
