from environments.gym_super_mario_bros import SuperMarioBrosEnv
import numpy as np

class SuperMario_Env(SuperMarioBrosEnv):
    def __init__(self, world, stage, version=0, scale=False):
        _ROM_MODES = ['vanilla', 'downsample', 'pixel', 'rectangle']
        target = (world, stage)
        rom_mode = _ROM_MODES[version]
        self.scale = scale
        super().__init__(rom_mode=rom_mode, lost_levels=False, target=target)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.scale:
            obs = np.uint8(obs * 255.0)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        if self.scale:
            obs = np.uint8(obs * 255.0)
        return obs




if __name__ == "__main__":
    import cv2
    from nes_py.wrappers import JoypadSpace
    from environments.gym_super_mario_bros.actions import RIGHT_ONLY
    world = 2
    stage = 1
    version = 0
    env = JoypadSpace(SuperMario_Env(world, stage, version), RIGHT_ONLY)
    obs = env.reset()
    import numpy as np

    for i in range(100000):
        obs, rewards, dones, info = env.step(env.action_space.sample())
        # cv2.imshow('obs', np.uint8(obs ))
        cv2.imshow('obs', cv2.cvtColor(np.uint8(obs * 255.0), cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(5)

        if dones:
            env.reset()

        if k == ord('b'):
            env.backup()

        if k == ord('o'):
            env.reset()

        if k == ord('r'):
            env.restore()

        if k == ord('q'):
            break



