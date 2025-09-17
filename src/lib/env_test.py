import gymnasium as gym
import random
import numpy as np

class TestEnv(gym.Env):
    def __init__(self):
        #super().__init__()
        self.battery_capacity = 0
        self.action_space = gym.spaces.Discrete(2)  # Example action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        self.steps = 0

    def __get_obs(self):
        return np.asarray([self.battery_capacity/100])
    
    def reset(self, *, seed = None, options = None):
        #super().reset(seed=seed, options=options)
        self.battery_capacity = 0  # Reset battery capacity to a random value
        self.steps = 0
        return self.__get_obs(), {}
    
    def step(self, action):
        reward = 0
        if action == 1: # Se la usi si scarica
            self.battery_capacity -= 1
            if self.battery_capacity <= 0:
                reward = -1 #Se scarica reward negativa
            else:
                reward = 1 #Se usata reward positiva
        else: #Se non la usi si rircarica
            self.battery_capacity += 1


        self.battery_capacity = max(0, min(self.battery_capacity, 100))

        self.steps += 1
         
        return self.__get_obs(), reward, False, self.steps>=300, {}
