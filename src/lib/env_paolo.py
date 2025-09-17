import numpy as np

ENCOURAGE_PRODUCTIVITY= -0.32 #penalty when being in idle
POS_REWARD= 1.0
NEG_REWARD= -1.0

import csv 
from datetime import datetime
import gymnasium as gym
import random

class Solar:
    def __init__(self, csv_path: str, scale_factor: float = 1.0, interval_s: int = 300):
        self.interval_s = interval_s  # 300 seconds per slot
        self.energy_kwh_vector = []

        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)

            # Prepare raw time → power (W) series
            first_dt = None
            raw_power_series = []

            for line in reader:
                try:
                    dt = datetime.fromisoformat(line[26])
                    if first_dt is None:
                        first_dt = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                    delta = dt - first_dt
                    time_s = int(delta.total_seconds())
                    power_w = float(line[12]) * scale_factor  # GTI column
                except Exception:
                    continue  # Skip malformed lines

                raw_power_series.append((time_s, power_w))

        # Fill energy_kwh_vector slot by slot
        max_time = raw_power_series[-1][0]
        num_slots = (max_time // interval_s) + 1
        self.energy_kwh_vector = [0.0] * num_slots

        idx = 0
        for slot in range(num_slots):
            slot_start = slot * interval_s
            slot_end = slot_start + interval_s
            energy_ws = 0.0

            while idx < len(raw_power_series) and raw_power_series[idx][0] < slot_end:
                time_s, power_w = raw_power_series[idx]
                energy_ws += power_w * interval_s  # assume constant power over the interval
                idx += 1

            self.energy_kwh_vector[slot] = energy_ws / 3_600_000  # Convert W·s to kWh

    def get_solar_energy(self, timestep: int) -> float:
        """Returns energy in kWh for 5-minute slot corresponding to timestep."""
        if 0 <= timestep < len(self.energy_kwh_vector):
            return self.energy_kwh_vector[timestep]
        return 0.0  # Out of bounds = no energy


class BatteryEnv(gym.Env):
    def __init__(self):
        self.solar = Solar(csv_path="../solcast2024.csv", scale_factor=1.0)
#        self.base_load_energy_ma = 50
#        self.full_load_energy_ma = 1000
        self.init_battery_percentage = 0.5
#        self.device_nominal_voltage_v = 5
#        self.battery_max_capacity_mah = 3300
#        self.base_load_energy_w = (self.base_load_energy_ma/1000)*self.device_nominal_voltage_v
#        self.full_load_energy_w = (self.full_load_energy_ma/1000)*self.device_nominal_voltage_v
#        self.battery_nominal_voltage_v = 3.7
        self.battery_max_capacity_kwh = 1 #self.battery_nominal_voltage_v*(self.battery_max_capacity_mah/1000)
        self.step_size_s = 1  # 5 minutes in seconds
        self.max_steps = self.step_size_s * 576 * 4 #un numero ragionevole per ora
        self.consume_cost = 0.29 # sono wh
        self.observation_space = gym.spaces.Box(0,1,[1],np.float32)
        self.action_space = gym.spaces.Discrete(2)

        self.reset()

    def reset(self,seed=None):
        self.battery = self.battery_max_capacity_kwh*random.randint(3,8)/10
        self.sim_time = 120  # Seconds from start of year
        return np.array([self.battery], dtype=np.float32), {}

    def step(self, action):
        done = False
        reward = 0.0

        # Get solar power in Watts → convert to kWh for 5 minutes
        recharge = self.solar.get_solar_energy(self.sim_time)*10 #recharge non si in che unità di misura sia ora
        #import pdb; pdb.set_trace()
        self.battery = min(self.battery_max_capacity_kwh, self.battery + recharge)
        self
        if action == 1:
            if self.battery >= self.consume_cost:
                self.battery -= self.consume_cost
                reward = POS_REWARD
            else:
                self.battery -= self.consume_cost
                reward = NEG_REWARD  # 
                done = True
        else:
            reward = ENCOURAGE_PRODUCTIVITY  # Encourage productivity

        self.sim_time += self.step_size_s
        if self.sim_time >= self.max_steps:
            done = True

        return np.array([self.battery], dtype=np.float32), reward, done, False, {"recharge": recharge, "cons":action}