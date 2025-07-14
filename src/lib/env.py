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
    def __init__(self, csv_solar_data_path: str, step_s: int, stop_on_full_battery: bool = False, discrete_action: bool = False, discrete_state: bool = False, truncate_alter_d: int = 30, incentive_factor:float=0.3):
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
            self.observation_space = gym.spaces.MultiDiscrete([5, 5, 2])
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

    def reset(self, *, seed=None, options=None):
        # Reset with random battery percentage
        if options is not None and options["norandom"] == True:
            self.device.reset(battery_percentage=0.5)
            self.boot_time_s = 5*60
        else:
            self.device.reset(battery_percentage=rand.randrange(2, 8)/10)
            self.boot_time_s = rand.randint(5*60, 60*60*24*300)  # Random day from 0 to 300
        self.time_s = self.boot_time_s
        obs = self.__get_obs()
        info = self.__get_info()
        return obs, info

    def step(self, action):  # NON VA BENE CHIAMARLO OGNI TOT SECONDI, MA OGNI FINE INFERENZA
        self.number_of_steps+=1
        if action==1 or (type(action)==list and action[0]==1):
            self.number_of_high+=1
        else:
            self.number_of_low +=1
        # Apply action
        if self.discrete_action:
            self.device.set_processing_rate(action)
        else:
            self.device.set_processing_rate(action[0])

        # Update internal device data
        self.device.update(self.time_s)

        # Terminate if battery is discharged or chaged at 100%
        # or (self.stop_on_full_battery and self.device.get_battery_percentage() == 1)
        terminated = self.device.get_battery_percentage() == 0
        if terminated:
            print(f"Battery discarged at time {self.time_s} s -> {self.get_human_uptime()}")
        # terminated |= self.time_s > 60*60*24*3

        # Get action effects
        # observation = self.__get_obs()

        # Compute reward
        # normalized_energy_consumption_w = self.device.get_energy_consumption_w()/(self.device.base_load_energy_w+self.device.full_load_energy_w)
        # OBJ: Incentiva processamento e disincentiva batteria scarica

        # reward = -abs(0.5-self.device.get_battery_percentage())*normalized_energy_consumption_w
        # NO! Quanto tutto è carico, la prima parte da rw=-0.5, la seconda parte dovrebbe incentivare a processare, invece andrà a 0 e quindi non processa

        # reward = self.device.get_battery_percentage()*normalized_energy_consumption_w
        # NO! perchè quando la batteria è scarica, batt=0.1, se processata ottiene qualcosa <0.1, se non processa ottiene 0 e quindi scarica la batteria

        # Processed images
        if self.get_uptime_s() > 0:
            # Non sarà mai uguale a 1 perchè passo del tempo a non processare
            processed_images = self.get_amount_processed_images(
            ) / (self.get_uptime_s()/self.device.task_duration_s)
        else:
            processed_images = 0
        
        # Energy
        energy_norm = self.device.get_energy_consumption_w(
        )/(self.device.base_load_energy_w + self.device.full_load_energy_w)

        battery_level = self.device.get_battery_percentage()  # ∈ [0, 1]
        reward = 1 - abs(energy_norm - battery_level) #Se batt_level scende sotto lo 0.5, sono incentivato a non fare niente
        
        #Reward 2
        # reward = 0 #Only second part
        if energy_norm > 0.9: #energy norm € (Base_Load, FullLoad)
            reward += self.incentive_factor  # spinta a consumare
        wh_for_next_step = self.device.base_load_energy_w * s2h(2*self.step_s)
        if self.device.battery_current_capacity_wh < wh_for_next_step: #?Qual'è il valore ottimo?
            reward -= 1  # forte penalità se scarica troppo

        # Reward 3 MALE 169 immagini
        # wh_for_next_step = self.device.base_load_energy_w * s2h(self.incentive_factor*self.step_s)
        # min_batt_level = wh_for_next_step / self.device.battery_max_capacity_wh
        # b = 1 if battery_level > min_batt_level else battery_level
        # reward = 1-abs(energy_norm - b)

        #reward 4
        # if self.device.battery_current_capacity_wh < wh_for_next_step:
        #     reward =-1
        # else:
        #     reward = energy_norm

        #reward = -energy_norm
        # Reward 3
        # battery_drop = self.prev_battery - battery_level
        # self.prev_battery = battery_level
        # reward = battery_drop
        # reward -= 0.5 if battery_level < 0.1 else 0

        # Reward 4 NON FUNZIONA, tipo 176 immagini
        # if self.device.get_pv_production_normalized() > 0: #Day
        #     reward = energy_norm #Molto piccolo
        # else:  # Night
        #     reward = -battery_level

        #incentive_image_processing = self.incentive_factor*processed_images
        #reward = reward+incentive_image_processing

        # Go to the future
        self.time_s += self.step_s

        # Increase future data
        solar_current_w = self.solar.get_solar_w(self.time_s)
        self.device.set_pv_production_current_w(solar_current_w)

        """                
        if self.last_day_state == None:
            self.last_day_state = "day" if self.device.is_day else "night"
        if self.last_day_state == "day" and self.device.is_day == False:
            # Switch to night
            self.last_day_state = "night"
            print("Night")
            terminated = True
        elif self.last_day_state == "night" and self.device.is_day:
            self.last_day_state = "day"
            print("day")
            terminated = False
         """

        # Truncate when no more data
        truncated = solar_current_w < 0

        # Truncate after 30 days
        truncated |= self.time_s - self.boot_time_s > 60*60*24*self.truncate_after_d

        info = self.__get_info()
        info["TimeLimit.truncated"] = truncated and not terminated

        return self.__get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"At time {self.time_s} seconds, battery level: {self.device.get_battery_percentage()}%")

    def __get_obs(self):
        if self.discrete_state:
            battery_state = math.floor(
                10*self.device.get_battery_percentage()/5)  # 0-3
            day_slot = math.floor((self.time_s % 86400) / 86400.0 * 5)  # 0-9
            return np.array([battery_state, day_slot, self.device.get_pv_production_normalized() > 0])
        else:
            # panel_production_a = self.solar.get_solar_w(self.time_s+60*30)/12
            # future_panel_production_normalized = panel_production_a/2.31
            # x1, y1 = self.time_s, self.device.get_pv_production_normalized()
            # x2, y2 = self.time_s+60*60, future_panel_production_normalized
            # angle_rad = math.atan2(y2-y1, x2-x1)
            # angle_deg = math.degrees(angle_rad)

            return np.array(
                [
                    self.device.get_pv_production_normalized(),
                    # self.device.get_energy_consumption_w()/(self.device.base_load_energy_w + self.device.full_load_energy_w),
                    self.device.get_battery_percentage(),
                    ((self.time_s/3600) % 24) / 24.0,  # Seconds in a day
                    # future_panel_production_normalized,
                    # self.solar.get_solar_w(self.time_s-60*30)/12/2.31
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
