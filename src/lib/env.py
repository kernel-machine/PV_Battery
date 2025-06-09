import gymnasium as gym
from lib.device import Device
import numpy as np
from lib.solar import solar
import random as rand
import time
import math
# from filterpy.kalman import ExtendedKalmanFilter


class NodeEnv(gym.Env):
    def __init__(self, csv_solar_data_path: str, step_s: int, stop_on_full_battery: bool = False, discrete_action: bool = False, discrete_state: bool = False):
        self.device = Device(
            base_load_energy_ma=20,
            full_load_energy_ma=500,
            battery_max_capacity_mah=3300,
            task_duration_s=1,
            init_battery_percentage=rand.random(),
            processing_rate_force_update_after_s=60*5,  # 5 minutes
        )
        super().__init__()
        self.time_s = rand.randint(5*60, 60*60*24)  # Random second in the day
        self.step_s = step_s
        # GTI*Surface(m2)*efficiency
        self.solar = solar.Solar(csv_solar_data_path, scale_factor=0.18*0.28)
        self.device.set_pv_production_current_ma(
            self.solar.get_solar_a(self.time_s))
        self.last_solar = self.device.get_pv_production_normalized()
        self.discrete_state = discrete_state
        if discrete_state:
            print("Using discrete state space")
            self.observation_space = gym.spaces.MultiDiscrete([5, 5, 2])
        else:
            self.observation_space = gym.spaces.Box(
                0, 1, [len(self.__get_obs())])
        self.discrete_action = discrete_action
        if discrete_action:
            print("Using discrete action space")
            self.action_space = gym.spaces.Discrete(2)
        else:
            self.action_space = gym.spaces.Box(0, 1, [1])
        self.stop_on_full_battery = stop_on_full_battery
        # self.ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
        # self.ekf.x = np.array([self.solar.get_solar(self.time_s)])  # Initial state

    def reset(self, *, seed=None, options=None):
        # print("Resetting environment")
        self.device.reset(rand.randrange(2,8)/10)  # Reset with random battery percentage
        obs = self.__get_obs()
        self.time_s = rand.randint(5*60, 60*60*24*7)
        return obs, self.__get_info()

    def step(self, action):  # NON VA BENE CHIAMARLO OGNI TOT SECONDI, MA OGNI FINE INFERENZA
        self.time_s += self.step_s
        if self.discrete_action:
            self.device.set_processing_rate(action)
        else:
            self.device.set_processing_rate(action[0])
        solar_current_ma = (self.solar.get_solar_a(
            self.time_s)/12)*1000  # I = W/V
        self.device.set_pv_production_current_ma(solar_current_ma)
        self.device.update(self.time_s)

        terminated = solar_current_ma < 0 or self.device.get_battery_percentage() == 0  or (self.device.get_battery_percentage() == 1 and self.stop_on_full_battery)
        nA1Sec = (self.device.last_energy_used_nah/self.step_s)
        max_nA1Sec = ((self.device.full_load_energy_na +
                      self.device.base_load_energy_na)/self.step_s)
        # if self.device.get_battery_percentage() < 0.05:
        #    reward = -10
        # elif self.device.get_battery_percentage() < 0.33:
        # reward = -nA1Sec / max_nA1Sec  # Try to reduce the energy consumption
        # else:
        #    reward = self.device.processing_rate*10*(1-self.device.get_battery_percentage()) # Reward for processing rate, penalize for low battery
        
        battery_percentage = self.device.get_battery_percentage()
        LOWER_BOUND = 0.2
        UPPER_BOUND = 0.8
        if battery_percentage < LOWER_BOUND:
            reward = -2*(LOWER_BOUND-battery_percentage)
        elif battery_percentage > UPPER_BOUND:
            reward = -2*(battery_percentage-UPPER_BOUND)
        else:
            reward = 0
        if terminated:
            print(
                f"Terminated after {self.time_s} seconds, with a battery level: {self.device.get_battery_percentage()*100}%")

        truncated = self.time_s > 60*60*24*30  # Reset after 1 month
        observation = self.__get_obs()
        info = self.__get_info()
        info["TimeLimit.truncated"] = truncated and not terminated
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"At time {self.time_s} seconds, battery level: {self.device.get_battery_percentage()}%")

    def __get_obs(self):
        if self.discrete_state:
            battery_state = math.floor(10*self.device.get_battery_percentage()/5)  # 0-3
            day_slot = math.floor((self.time_s % 86400) / 86400.0 * 5)  # 0-9
            return np.array([battery_state, day_slot, self.device.get_pv_production_normalized()>0])
        else:
            panel_production_a = self.solar.get_solar_a(self.time_s+60*30)/12
            future_panel_production_normalized = panel_production_a/2.31
            x1, y1 = self.time_s, self.device.get_pv_production_normalized()
            x2, y2 = self.time_s+60*60, future_panel_production_normalized
            angle_rad = math.atan2(y2-y1, x2-x1)
            angle_deg = math.degrees(angle_rad)

            return np.array(
                [
                    self.device.get_pv_production_normalized(),
                    self.device.get_energy_consumption_ma()/((self.device.base_load_energy_na +
                                                              self.device.full_load_energy_na))/1000,
                    self.device.get_battery_percentage(),
                    (self.time_s % 86400) / 86400.0,  # Seconds in a day
                    future_panel_production_normalized,
                    self.solar.get_solar_a(self.time_s-60*30)/12/2.31
                    # angle_deg,

                    # self.device.last_wasted_energy_mah/self.device.last_harvested_energy_nah if self.device.last_harvested_energy_nah>0 else 0
                    # self.get_angular_coefficient(steps_s=30*60) #Between 30 minutes
                    # self.device.processing_rate,
                ],
                dtype=np.float32,
            )

    def __get_info(self):
        return {
            "pv_solar": self.device.get_pv_production_normalized(),
            "current": self.device.get_energy_consumption_ma()/((self.device.base_load_energy_na +
                                                                self.device.full_load_energy_na))/1000,
            "battery": self.device.get_battery_percentage(),
        }

    def get_angular_coefficient(self, steps_s: int) -> float:
        intervals = steps_s//self.step_s
        if len(self.device.energy_harvested_ma) < 2:
            return 0
        else:
            number_of_items = len(self.device.energy_harvested_ma)
            p2_index = number_of_items - 1
            p1_index = max(0, p2_index - intervals)
            p1_x, p1_y = p1_index, self.device.energy_harvested_ma[p1_index]
            p2_x, p2_y = p2_index, self.device.energy_harvested_ma[p2_index]
            m = (p2_y-p1_y)/(p2_x-p1_x)
            if m > 0:
                return 1
            elif m < 0:
                return -1
            else:
                return 0

    def get_uptime_s(self) -> int:
        return self.time_s

    def get_human_uptime(self) -> str:
        days = self.get_uptime_s() // 86400
        output = f"{days} days, "
        output += time.strftime("%H hours, %M minutes, %S seconds",
                                time.gmtime(self.get_uptime_s()))
        return output

    def get_amount_processed_images(self) -> int:
        return self.device.total_processed_images
