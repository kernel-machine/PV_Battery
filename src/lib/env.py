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
                 battery_wh:float = 6.6*3.7,
                 device_idle_energy_w = 5*0.05,
                 device_full_energy_w = 5,
                 seed = None,
                 use_solar:bool = False,
                 use_month:bool = False,
                 use_hour:bool = False,
                 use_day:bool = False
                   ):
        self.use_solar = use_solar
        self.use_month = use_month
        self.use_hour = use_hour
        self.use_day = use_day
    
        panel_area_m2 = 0.55*0.51 #m2
        efficiency = 0.1426
        max_power_w = 40 #W
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True)
        self.battery_max_j = battery_wh*3600
        self.battery_curr_j = self.battery_max_j*0.5
        self.time_s = 5*60
        self.observation_space = gym.spaces.Box(0.0, 1.0, [len(self.__get_obs())])
        self.action_space = gym.spaces.Discrete(2)
        self.step_size_s = step_s
        self.device_idle_energy_w = device_idle_energy_w
        self.device_full_energy_w = device_full_energy_w
        self.processed_images = 0
        self.harvested_energy_j = 0

        if seed is not None: 
            random.seed(seed)
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage, used for validation
        if options is not None and options["norandom"] == True:
            self.boot_time_s = (60*5)#*rand.randint(1,30)
            self.battery_curr_j = self.battery_max_j*0.5
        else:
            random_day_s = 60*60*24*rand.randint(1,300)
            random_hour_s = 0#60*60*random.randint(0,23)
            self.boot_time_s = (60*60*5)+random_day_s+random_hour_s
            self.battery_curr_j = self.battery_max_j*(random.randint(4,9)/10)

        self.time_s = self.boot_time_s
        self.processed_images = 0
        self.harvested_energy_j = self.battery_curr_j
        obs = self.__get_obs()
        return obs, {}

    def step(self, action):
        return self.step_2(action)
        #return self.step_work(action)
    
    def step_2(self, action):
        # Set solar data
        solar_power_w = self.solar.get_solar_w(self.time_s) #-1 if end of data
        solar_energy_j = max(0,solar_power_w*self.step_size_s)

        #image_energy_j = (self.device_full_energy_w if action==1 else self.device_idle_energy_w)*self.step_size_s
        processing_energy_j = self.device_full_energy_w*self.step_size_s
        idle_energy_j = self.device_idle_energy_w*self.step_size_s

        self.battery_curr_j += solar_energy_j
        batt_perc = self.battery_curr_j/self.battery_max_j
        if action == 1:
            if self.battery_curr_j > processing_energy_j:
                reward = 1+batt_perc
                self.processed_images+=1
            else:
                reward = -1 #-5 
            self.battery_curr_j -= processing_energy_j
        else:
            self.battery_curr_j -= idle_energy_j
            reward = -2*batt_perc #-0.4
        
        self.battery_curr_j = max(0,min(self.battery_max_j, self.battery_curr_j))
        datetime = self.solar.get_datetime(self.time_s)
        self.time_s += self.step_size_s
        terminated = self.get_uptime_s() >= 60*60*24*30 or solar_power_w < 0
        done = self.battery_curr_j<=0

        return self.__get_obs(), reward, done, terminated, {"recharge":solar_power_w/40, "cons":action, "time":datetime}
    
    def step_work(self, action):  # Funziona ma non arriva mai al 100%

        # Set solar data
        solar_power_w = self.solar.get_solar_w(self.time_s) #-1 if end of data
        solar_energy_j = max(0,solar_power_w*self.step_size_s)

        #image_energy_j = (self.device_full_energy_w if action==1 else self.device_idle_energy_w)*self.step_size_s
        processing_energy_j = (self.device_full_energy_w*self.step_size_s)
        idle_energy_j = (self.device_idle_energy_w*self.step_size_s)

        self.battery_curr_j = min(self.battery_max_j, self.battery_curr_j + solar_energy_j)
        if action == 1:
            if self.battery_curr_j > processing_energy_j:
                reward = 1
                self.processed_images+=1
            else:
                reward = -2
            self.battery_curr_j -= processing_energy_j
        else:
            self.battery_curr_j -= idle_energy_j
            reward = 0  # Encourage productivity

        self.time_s += self.step_size_s
        terminated = self.get_uptime_s() >= 60*60*24*7 or solar_power_w < 0
        done = self.battery_curr_j<=0

        return self.__get_obs(), reward, done, terminated, {"recharge":solar_power_w/40, "cons":action}

    def render(self, mode="human"):
        pass

    def __get_obs(self):
        arr = [
            self.battery_curr_j/self.battery_max_j,
        ]
        if self.use_solar:
            arr.append(self.solar.get_solar_w(self.time_s)/40)
        if self.use_month:
            arr.append(self.solar.get_datetime(self.time_s).month/12)
        if self.use_hour:
            #m = self.solar.get_datetime(self.time_s).minute #0-59
            h = self.solar.get_datetime(self.time_s).hour #0-23
            arr.append(h/23)
        if self.use_day:
            arr.append(self.solar.get_datetime(self.time_s).timetuple().tm_yday/366)
        return np.array(arr,dtype=np.float32)


    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

