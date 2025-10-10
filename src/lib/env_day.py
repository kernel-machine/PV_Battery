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
    SOLAR           = 1<<0
    MONTH           = 1<<1
    HOUR            = 1<<2
    DAY             = 1<<3
    NEXT_DAY        = 1<<4
    HUMIDITY        = 1<<5
    PRESSURE        = 1<<6
    CLOUD           = 1<<7
    SUN_REAL_PREDICTION  = 1<<8
    SUN_ESTIMATE_PREDICTION = 1<<9
    SUN_ESTIMATE_SINGLE_PREDICTION = 1<<10
    MINUTE = 1<<11

class EnvDay(gym.Env):
    def __init__(self, csv_solar_data_path: str,
                  step_s: int, 
                 battery_wh:float = 6.6*3.7,
                 device_idle_energy_w = 5*0.05,
                 device_full_energy_w = 5,
                 seed:int = 1234,
                 state_content:int = StateContent.SOLAR,
                 random_reset:bool = True,
                 terminated_days:int = 7,
                 forecast_time_m:int = 60,
                 prediction_accuracy:float = 1.0,
                 choose_forecast:bool = False
                   ):
        self.state_content = state_content
        self.random_reset = random_reset
        self.terminated_days = terminated_days
        self.forecast_time_m = forecast_time_m
        self.choose_forecast = choose_forecast
    
        panel_area_m2 = 0.55*0.51 #m2
        efficiency = 0.1426
        max_power_w = 40 #W
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=prediction_accuracy)
        self.battery_max_j = battery_wh*3600
        self.battery_curr_j = self.battery_max_j*0.5
        self.time_s = 5*60
        self.step_size_s = step_s
        if self.state_content & StateContent.NEXT_DAY:
            self.max_avg_day = max(self.solar.get_day_avg(x) for x in range(366))
        self.device_idle_energy_w = device_idle_energy_w
        self.device_full_energy_w = device_full_energy_w
        self.device_idle_energy_j = device_idle_energy_w*self.step_size_s
        self.device_full_energy_j = device_full_energy_w*self.step_size_s
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j
        self.observation_space = gym.spaces.Box(0.0, 1.0, [len(self.__get_obs()[0])])
        if choose_forecast:
            self.action_space = gym.spaces.Box(low=0,high=1,shape=(3,),dtype=float)
        else:
            self.action_space = gym.spaces.Discrete(2)


        self.rng = random.Random(seed)
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage, used for validation
        if (options is not None and options["norandom"] == True) or not self.random_reset:
            self.boot_time_s = 5*60#self.solar.get_next_sunrise(60*5)
            if (options is not None and "start_day" in options.keys()):
                self.boot_time_s += 24*60*60*options["start_day"]
            self.battery_curr_j = self.battery_max_j*0.5
        else:
            random_day_s = 60*60*24*self.rng.randint(1,30)
            random_hour_s = self.rng.randint(0,23)*60*60
            self.boot_time_s = random_day_s+random_hour_s#self.solar.get_next_sunrise(random_day_s)
            self.battery_curr_j = self.battery_max_j*(self.rng.randint(4,9)/10)

        self.time_s = self.boot_time_s
        self.processed_images = 0
        self.haversted_energy_j = self.battery_curr_j
        obs, fields = self.__get_obs()
        return obs, {"fields":fields}

    
    def step(self, action):
        def emphasize_diff_sigmoid(x, sharpness=10):
            return 1 / (1 + math.exp(-sharpness * (x - 0.5)))
        # Set solar data
        solar_power_w = self.solar.get_solar_w(self.time_s) #-1 if end of data
        solar_energy_j = max(0,solar_power_w*self.step_size_s)

        self.haversted_energy_j += solar_energy_j
        self.battery_curr_j += solar_energy_j

        if self.choose_forecast:
            process = 1 if action[0]>0.5 else 0
        else:
            process = action

        if process == 1:
            if self.battery_curr_j > self.device_full_energy_j:
                self.processed_images+=1
            self.battery_curr_j -= self.device_full_energy_j
        else:
            self.battery_curr_j -= self.device_idle_energy_j
        
        reward = (-self.battery_curr_j+self.battery_max_j)/self.battery_max_j
        self.battery_curr_j = max(0,min(self.battery_max_j, self.battery_curr_j))
        datetime = self.solar.get_datetime(self.time_s)
        efficiency = (self.device_full_energy_j*self.processed_images)/self.haversted_energy_j
        #reward = math.log10(efficiency)+1
        #reward = emphasize_diff_sigmoid(efficiency)
        batt_perc = self.battery_curr_j / self.battery_max_j
        #reward = (-1 if process == 0 else 1)*(batt_perc)
        #reward = -abs(batt_perc - process) + 0.3*process
        #reward = efficiency
        terminated = self.get_uptime_s() > 60*60*24*self.terminated_days

        #reward = -(abs(batt_perc-process)**2)#+(0.2*process)

        done = self.battery_curr_j<=0

        self.time_s += self.step_size_s

        if self.choose_forecast:
            when_m = int(action[1]*26*60)
            window_size_m = int(action[2]*24*60)
            obs, fields = self.__get_obs(when_m, window_size_m)
        else:
            obs, fields = self.__get_obs()
        info = {"fields":fields, "cons":process, "time":datetime, "forecast":(action[1:] if self.choose_forecast else 0)}
        #print("OBS",obs)
        return obs, reward, done, terminated, info

    def render(self, mode="human"):
        pass

    def __get_obs(self, when_m:int = 0, windows_size_m:int = 60)->tuple[np.array,list]:
        fields = [
            "Battery"
        ]
        arr = [
            self.battery_curr_j/self.battery_max_j
        ]
        max_steps = self.terminated_days*24*60/5
        
        # fields.append("Processed images")
        # arr.append(self.processed_images/max_steps)

        # fields.append("Harvested energy")
        # arr.append(self.haversted_energy_j/(max_steps*40*self.step_size_s))

        if self.state_content & StateContent.SOLAR:
            arr.append(self.solar.get_solar_w(self.time_s)/40)
            fields.append("Solar")
        if self.state_content & StateContent.MONTH:
            arr.append(self.solar.get_datetime(self.time_s).month/12)
            fields.append("Month")
        if self.state_content & StateContent.HOUR:
            #m = self.solar.get_datetime(self.time_s).minute #0-59
            h = self.solar.get_datetime(self.time_s).hour #0-23
            arr.append(h/23)
            fields.append("Hour")
        if self.state_content & StateContent.MINUTE:
            m = self.solar.get_datetime(self.time_s).minute #0-59
            arr.append(m/59)
            fields.append("Minute")
        if self.state_content & StateContent.DAY:
            arr.append(self.solar.get_datetime(self.time_s).timetuple().tm_yday/366)
            fields.append("Day")
        if self.state_content & StateContent.NEXT_DAY:
            day = self.solar.get_datetime(self.time_s).timetuple().tm_yday
            next_day_sun = self.solar.get_day_avg(day+1)/self.max_avg_day
            arr.append(next_day_sun)
            fields.append("Next day")
        if self.state_content & StateContent.SUN_REAL_PREDICTION:
            energy_j = self.solar.get_real_future_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            if windows_size_m > 0:
                energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                energy_j = 0
            arr.append(energy_j)
            fields.append("Sun real prediction")
        if self.state_content & StateContent.SUN_ESTIMATE_PREDICTION:
            lower_bound_energy_j, upper_bound_energy_j = self.solar.get_estimate_future_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            # Normalize
            if windows_size_m > 0:
                lower_bound_energy_j /= self.solar.max_power_w*windows_size_m*60
                upper_bound_energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                lower_bound_energy_j = 0
                upper_bound_energy_j = 0
            arr.append(lower_bound_energy_j)
            arr.append(upper_bound_energy_j)
            fields.append("Sun estimate prediction ub")
            fields.append("Sun estimate prediction lb")
        if self.state_content & StateContent.SUN_ESTIMATE_SINGLE_PREDICTION:
            energy_j = self.solar.get_estimate_future_single_prediction_j(self.time_s+when_m*60, self.step_size_s, windows_size_m)
            if windows_size_m > 0:
                energy_j /= self.solar.max_power_w*windows_size_m*60
            else:
                energy_j = 0
            arr.append(energy_j)
            fields.append("Energy prediction")
        if self.state_content & StateContent.PRESSURE:
            val = self.solar.get_info(self.time_s,"pressure")
            arr.append(val/1000)
            fields.append("Pressure")
        if self.state_content & StateContent.HUMIDITY:
            val = self.solar.get_info(self.time_s,"humidity")
            arr.append(val/100)
            fields.append("Humidity")
        if self.state_content & StateContent.CLOUD:
            val = self.solar.get_info(self.time_s,"cloud_opacity")
            arr.append(val/100)
            fields.append("Cloud opacity")
        return np.array(arr,dtype=np.float32),fields


    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

