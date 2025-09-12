import gymnasium as gym
from lib.device import Device
import numpy as np
from lib.solar import solar
import random as rand
import time
import math
from lib.utils import s2h
import random
from enum import IntEnum

# from filterpy.kalman import ExtendedKalmanFilter

class StateContent(IntEnum):
    SOLAR   = 1<<0
    MONTH   = 1<<1
    HOUR    = 1<<2
    DAY     = 1<<3
    FORECAST= 1<<4
    HUMIDITY= 1<<5
    PRESSURE= 1<<6
    CLOUD   = 1<<7

class EnvDay(gym.Env):
    def __init__(self, csv_solar_data_path: str,
                  step_s: int, 
                 battery_wh:float = 6.6*3.7,
                 device_idle_energy_w = 5*0.05,
                 device_full_energy_w = 5,
                 seed = None,
                 state_content:int = StateContent.SOLAR,
                 random_reset:bool = True,
                 terminated_days:int = 7,
                   ):
        self.state_content = state_content
        self.random_reset = random_reset
        self.terminated_days = terminated_days
    
        panel_area_m2 = 0.55*0.51 #m2
        efficiency = 0.1426
        max_power_w = 40 #W
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True)
        self.battery_max_j = battery_wh*3600
        self.battery_curr_j = self.battery_max_j*0.5
        self.time_s = 5*60
        self.max_avg_day = max(self.solar.get_day_avg(x) for x in range(366))
        self.observation_space = gym.spaces.Box(0.0, 1.0, [len(self.__get_obs())])
        self.action_space = gym.spaces.Discrete(2)
        self.step_size_s = step_s
        self.device_idle_energy_w = device_idle_energy_w
        self.device_full_energy_w = device_full_energy_w
        self.device_idle_energy_j = device_idle_energy_w*self.step_size_s
        self.device_full_energy_j = device_full_energy_w*self.step_size_s
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j

        if seed is not None: 
            random.seed(seed)
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage, used for validation
        if (options is not None and options["norandom"] == True) or not self.random_reset:
            self.boot_time_s = 5*60#self.solar.get_next_sunrise(60*5)
            if (options is not None and "start_day" in options.keys()):
                self.boot_time_s += 24*60*60*options["start_day"]
            self.battery_curr_j = self.battery_max_j*0.5
        else:
            random_day_s = 60*60*24*rand.randint(1,30)
            random_hour_s = rand.randint(0,23)*60*60
            self.boot_time_s = random_day_s+random_hour_s#self.solar.get_next_sunrise(random_day_s)
            self.battery_curr_j = self.battery_max_j*(random.randint(4,9)/10)

        self.time_s = self.boot_time_s
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j
        obs = self.__get_obs()
        return obs, {}

    
    def step(self, action):
        def emphasize_diff_sigmoid(x, sharpness=10):
            return 1 / (1 + math.exp(-sharpness * (x - 0.5)))
        # Set solar data
        solar_power_w = self.solar.get_solar_w(self.time_s) #-1 if end of data
        solar_energy_j = max(0,solar_power_w*self.step_size_s)

        self.haversted_energy_j += solar_energy_j
        self.battery_curr_j += solar_energy_j

        if action == 1:
            if self.battery_curr_j > self.device_full_energy_j:
                self.processed_images+=1
            self.battery_curr_j -= self.device_full_energy_j
        else:
            self.battery_curr_j -= self.device_idle_energy_j
        
        self.battery_curr_j = max(0,min(self.battery_max_j, self.battery_curr_j))
        datetime = self.solar.get_datetime(self.time_s)
        efficiency = (self.device_full_energy_j*self.processed_images)/self.haversted_energy_j
        #reward = math.log10(efficiency)+1
        reward = emphasize_diff_sigmoid(efficiency)
        # Incentive to discarge the battery
        #reward += (1-action)*(1-(self.battery_curr_j/self.battery_max_j))
        #is_sunrise = self.solar.is_sunrise(self.time_s,self.step_size_s)
        terminated = self.get_uptime_s() > 60*60*24*self.terminated_days
        done = self.battery_curr_j<=0

        self.time_s += self.step_size_s

        obs = self.__get_obs()
        info = {"recharge":solar_power_w/40, "cons":action, "time":datetime}

        return obs, reward, done, terminated, info

    def render(self, mode="human"):
        pass

    def __get_obs(self):
        arr = [
            self.battery_curr_j/self.battery_max_j,
        ]
        if self.state_content & StateContent.SOLAR:
            arr.append(self.solar.get_solar_w(self.time_s)/40)
        if self.state_content & StateContent.MONTH:
            arr.append(self.solar.get_datetime(self.time_s).month/12)
        if self.state_content & StateContent.HOUR:
            m = 0#self.solar.get_datetime(self.time_s).minute #0-59
            h = self.solar.get_datetime(self.time_s).hour #0-23
            arr.append(h/23)
        if self.state_content & StateContent.DAY:
            arr.append(self.solar.get_datetime(self.time_s).timetuple().tm_yday/366)
        if self.state_content & StateContent.FORECAST:
            day = self.solar.get_datetime(self.time_s).timetuple().tm_yday
            next_day_sun = 0# self.solar.get_day_avg(day+1)/self.max_avg_day
            arr.append(next_day_sun)
        if self.state_content & StateContent.PRESSURE:
            val = 0#self.solar.get_info(self.time_s,"pressure")
            arr.append(val/1000)
        if self.state_content & StateContent.HUMIDITY:
            val = self.solar.get_info(self.time_s,"humidity")
            arr.append(val/100)
        if self.state_content & StateContent.CLOUD:
            val = 0#self.solar.get_info(self.time_s,"cloud_opacity")
            arr.append(val/100)
        return np.array(arr,dtype=np.float32)


    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

