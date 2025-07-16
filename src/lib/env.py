import gymnasium as gym
from lib.device import Device
import numpy as np
from lib.solar import solar
import random as rand
import time
import math
from lib.utils import s2h

# from filterpy.kalman import ExtendedKalmanFilter


class NodeEnv(gym.Env):
    def __init__(self, csv_solar_data_path: str, step_s: int, stop_on_full_battery: bool = False, discrete_action: bool = False, discrete_state: bool = False, truncate_alter_d: int = 30, incentive_factor:float=0.3, episode_end_on_sunrise: bool = False):
        self.device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=6600,
            battery_nominal_voltage_v=3.7,
            task_duration_s=1,
            init_battery_percentage=rand.random(),
            processing_rate_force_update_after_s=60*5,  # 5 minutes
        )
        super().__init__()

        # Timing
        self.time_s = rand.randint(5*60, 60*60*24)  # Random second in the day
        self.boot_time_s = self.time_s
        self.step_s = step_s
        self.truncate_after_d = truncate_alter_d
        self.number_of_steps = 0
        self.number_of_high = 0
        self.number_of_low = 0

        self.battery = 0.5

        # Solar stuff
        # W = GTI*Surface(m2)*efficiency
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=0.28*0.18)
        self.device.set_pv_production_current_w(
            self.solar.get_solar_w(self.time_s))
        self.last_solar = self.device.get_pv_production_normalized()

        # State
        self.discrete_state = discrete_state
        if discrete_state:
            print("Using discrete state space")
            self.observation_space = gym.spaces.MultiDiscrete([5])
        else:
            self.observation_space = gym.spaces.Box(
                0, 1, [len(self.__get_obs())])

        # Action
        self.discrete_action = discrete_action
        if discrete_action:
            print("Using discrete action space")
            self.action_space = gym.spaces.Discrete(2)
        else:
            print("Using continuos action space")
            self.action_space = gym.spaces.Box(0, 1, [1])

        # Training stuff
        self.stop_on_full_battery = stop_on_full_battery
        self.incentive_factor = incentive_factor
        # self.ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
        # self.ekf.x = np.array([self.solar.get_solar(self.time_s)])  # Initial state
        self.last_day_state = None
        self.episode_end_on_sunrise = episode_end_on_sunrise

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage, used for validation
        #if True or options is not None and options["norandom"] == True:
        self.boot_time_s = (60*60*7)
        self.device.reset(self.boot_time_s,battery_percentage=0.5)
        #else:
            # self.boot_time_s = rand.randint(5*60, 60*60*24*300)
            # #self.boot_time_s = self.solar.get_next_sunrise(self.boot_time_s)
            # self.device.reset(self.boot_time_s, battery_percentage=rand.randrange(2, 8)/10)
        self.time_s = self.boot_time_s
        self.device.update(self.time_s)

        self.battery = 0.5

        obs = self.__get_obs()
        info = self.__get_info()
        return obs, info

    def step(self, action):  # NON VA BENE CHIAMARLO OGNI TOT SECONDI, MA OGNI FINE INFERENZA
        done = False
        reward = 0.0

        # Set solar data
        solar_current_w = self.solar.get_solar_w(self.time_s)/40
        self.device.set_pv_production_current_w(solar_current_w)
        image_consumption = 0.1
        energy_consumption = -image_consumption

        self.battery = min(1.0, self.battery + solar_current_w)
        if action == 1:
            if self.battery > image_consumption:
                self.battery -= image_consumption
                reward = 1
            else:
                self.battery -= image_consumption
                reward = -1  # 
                done = True
            energy_consumption = image_consumption
        else:
            reward = -0.32  # Encourage productivity
    
        self.battery -= image_consumption/100
        # Truncate when no more data
        #truncated = solar_current_w < 0

        # Truncate after 30 days
        self.time_s += self.step_s
        done |= self.get_uptime_s() >= (self.step_s*4*576)#self.truncate_after_d or (self.episode_end_on_sunrise and self.device.is_sunrise(self.time_s))

        #terminated |= self.time_s > 1000

        return self.__get_obs(), reward, done, False, {"recharge":solar_current_w, "cons":energy_consumption}

    def render(self, mode="human"):
        pass

    def __get_obs(self):
        return np.array(
            [
                # self.device.get_pv_production_normalized(),
                # self.device.get_energy_consumption_w()/(self.device.base_load_energy_w + self.device.full_load_energy_w),
                #self.device.get_battery_percentage(),
                self.battery,
                #((self.time_s/3600) % 24) / 24.0,  # Seconds in a day
                # future_panel_production_normalized,
                # self.solar.get_solar_w(self.time_s-60*30)/12/2.31
                # angle_deg,
            ],
            dtype=np.float32,
        )

    def __get_info(self):
        return {
            "pv_solar": self.device.get_pv_production_normalized(),
            "current": self.device.get_energy_consumption_w()/((self.device.base_load_energy_w +
                                                                self.device.full_load_energy_w))/1000,
            "battery": self.device.get_battery_percentage(),
        }

    def get_angular_coefficient(self, steps_s: int) -> float:
        intervals = steps_s//self.step_s
        if len(self.device.energy_harvested_w) < 2:
            return 0
        else:
            number_of_items = len(self.device.energy_harvested_w)
            p2_index = number_of_items - 1
            p1_index = max(0, p2_index - intervals)
            p1_x, p1_y = p1_index, self.device.energy_harvested_w[p1_index]
            p2_x, p2_y = p2_index, self.device.energy_harvested_w[p2_index]
            m = (p2_y-p1_y)/(p2_x-p1_x)
            if m > 0:
                return 1
            elif m < 0:
                return -1
            else:
                return 0

    def get_uptime_s(self) -> int:
        return self.time_s-self.boot_time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

    def get_amount_processed_images(self) -> int:
        return self.device.total_processed_images

if __name__=="__main__":
    env = NodeEnv(
        "solcast2024.csv",
        step_s=60,
        stop_on_full_battery=False,
        discrete_action=True,
        discrete_state=False,
        episode_end_on_sunrise=True
    )
    env.reset()
    print("Batt",env.device.get_battery_percentage())