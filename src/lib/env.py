import gymnasium as gym
from lib.device import Device
import numpy as np
from lib.solar import solar
import random as rand
import time
import math
from lib.utils import s2h
import random

# from filterpy.kalman import ExtendedKalmanFilter

class NodeEnv(gym.Env):
    def __init__(self, csv_solar_data_path: str,
                  step_s: int, 
                 battery_wh:float = 3.3*3.7,
                 device_idle_energy_w = 5*0.05,
                 device_full_energy_w = 5,
                 seed = None
                   ):
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=0.28*0.18)
        self.battery_max_j = battery_wh*3600
        self.battery_curr_j = self.battery_max_j*0.5
        self.time_s = 5*60
        self.observation_space = gym.spaces.Box(0, 1, [len(self.__get_obs())])
        self.action_space = gym.spaces.Discrete(2)
        self.step_size_s = step_s
        self.device_idle_energy_w = device_idle_energy_w
        self.device_full_energy_w = device_full_energy_w
        self.processed_images = 0
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage, used for validation
        if options is not None and options["norandom"] == True:
            self.boot_time_s = (60*60*5)#*rand.randint(1,30)
            self.battery_curr_j = self.battery_max_j*0.5
        else:
            self.boot_time_s = (60*60*5)+24*rand.randint(1,200)
            self.battery_curr_j = self.battery_max_j*(random.randint(3,8)/10)
            # self.boot_time_s = rand.randint(5*60, 60*60*24*300)
            # #self.boot_time_s = self.solar.get_next_sunrise(self.boot_time_s)
            # self.device.reset(self.boot_time_s, battery_percentage=rand.randrange(2, 8)/10)
        self.time_s = self.boot_time_s
        self.processed_images = 0
        obs = self.__get_obs()
        return obs, {}

    def step(self, action):  # NON VA BENE CHIAMARLO OGNI TOT SECONDI, MA OGNI FINE INFERENZA
        done = False

        # Set solar data
        solar_current_w = self.solar.get_solar_w(self.time_s)
        solar_current_j = solar_current_w*self.step_size_s

        image_energy_j = (self.device_full_energy_w if action==1 else self.device_idle_energy_w)*self.step_size_s
        self.battery_curr_j = min(self.battery_max_j, self.battery_curr_j + solar_current_j)
        if action == 1:
            if self.battery_curr_j > image_energy_j:
                reward = 1
                self.processed_images+=1
            else:
                reward = -2 
        else:
            reward = -0  # Encourage productivity
        self.battery_curr_j -= image_energy_j #TODO per questo la batteria non sta mai al 100%
        done |= self.battery_curr_j<=0

        self.time_s += self.step_size_s
        terminated = self.time_s >= 60*60*24*7#
        done |= solar_current_w < 0

        return self.__get_obs(), reward, done, terminated, {"recharge":solar_current_w/40, "cons":action}

    def render(self, mode="human"):
        pass

    def __get_obs(self):
        return np.array(
            [
                # self.device.get_pv_production_normalized(),
                # self.device.get_energy_consumption_w()/(self.device.base_load_energy_w + self.device.full_load_energy_w),
                self.battery_curr_j/self.battery_max_j,
                self.solar.get_datetime(self.time_s).month/12,
                self.solar.get_datetime(self.time_s).hour/23,
                #(self.time_s % 3600*24)/24
            ],
            dtype=np.float32,
        )


    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

