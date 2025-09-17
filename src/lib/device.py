import matplotlib.pyplot as plt
from lib.utils import s2h,h2s
from enum import Enum

class ProcessingState(Enum):
    RUNNING = 1
    STOPPED = 2

class Device:
    def __init__(self,
                 base_load_energy_ma: int,
                 full_load_energy_ma: int,
                 battery_max_capacity_mah: int,
                 battery_nominal_voltage_v: float,
                 task_duration_s: int,
                 device_nominal_voltage_v: float = 5,
                 init_battery_percentage: float = 0.5,
                 processing_rate_force_update_after_s: int = 0,):

        # Device specific parameters
        self.base_load_energy_w = (base_load_energy_ma/1000)*device_nominal_voltage_v
        self.full_load_energy_w = (full_load_energy_ma/1000)*device_nominal_voltage_v
        self.battery_nominal_voltage_v = battery_nominal_voltage_v
        self.battery_max_capacity_wh = self.battery_nominal_voltage_v*(battery_max_capacity_mah/1000)
        self.battery_current_capacity_wh = self.battery_max_capacity_wh*init_battery_percentage  # Battery at 50%
        self.discrete_processing:bool = True

        # PV specifications
        self.pv_instant_production_na = 0

        # Algoritm parameters
        self.processing_rate: float = 0
        self.target_processing_rate: float = 0
        self.next_inference_time_s = 0
        self.last_update_s = 0  # Used for energy estimation
        self.task_start_time_s = 0
        self.task_duration_s = task_duration_s
        self.processing_rate_force_update_after_s = processing_rate_force_update_after_s
        self.processing_state:int = 0
        self.total_processing_time_s = 0

        # Store statistics
        self.times = []
        self.battery_levels_percentage = []
        self.energy_consumption_w = []
        self.energy_harvested_w = []
        self.processing_rates = []
        self.total_processed_images = 0
        self.total_processed_images_last_time = 0
        self.total_consumed_energy_wh = 0
        self.total_produced_energy_wh = 0
        self.total_wasted_energy_wh = 0

        self.last_wasted_energy_wh = 0
        self.last_harvested_energy_nah = 0
        self.last_energy_used_wh = 0


        self.local_energy_counter_wh = 0
        self.maximum_rates = []
        self.is_day = False
        self.is_maximum_reached = False
        self.plot_added = False
        self.maximum_battery_time_s = 0
        self.day_start_time_s = 0
        self.day_end_time_s = 0

        self.max_production = 0

    # Update the instant production of the PV
    def set_pv_production_current_w(self, pv_w: float):
        self.pv_instant_w = pv_w

    def get_pv_production_normalized(self) -> float:
        return self.pv_instant_w / 40

    def set_processing_rate(self, processing_rate: float):
        self.processing_rate = max(0.0, min(processing_rate, 1.0))

    def update(self, time_s: float):
        # print("Check")
        if time_s < self.last_update_s:
            return
        
        update_delta_time_h = s2h(time_s-self.last_update_s)
        update_delta_time_s = h2s(update_delta_time_h)
        self.last_update_s = time_s

        if self.processing_rate == 1 and self.processing_state == 0:
            self.task_start_time_s = time_s
            self.processing_state = 1
        elif self.processing_state == 1 and self.processing_rate == 0:
            self.total_processed_images += update_delta_time_s//self.task_duration_s
            self.processing_state = 0
        elif self.processing_state == 1 and self.processing_rate == 1:
            self.total_processed_images += update_delta_time_s//self.task_duration_s
        elif self.processing_state == 0 and self.processing_rate == 0:
            pass
            # Conta da quanto tempo stai processando e conta le immagini
        """
        if (time_s > self.next_inference_time_s or time_s==0) and self.processing_rate > 0:
            # Start a new task
            if self.processing_state == 0:
                self.task_start_time_s = time_s
                self.processing_state = 1

            wait_time_s = (self.task_duration_s) * \
                (1-self.processing_rate)/self.processing_rate

            if self.processing_rate_force_update_after_s > 0:  # To avoid infinite wait time
                self.next_inference_time_s = time_s + \
                    min(wait_time_s, self.processing_rate_force_update_after_s)
            else:
                self.next_inference_time_s = time_s + wait_time_s

            self.total_processed_images += (time_s-self.task_start_time_s)//self.task_duration_s
        elif time_s < self.next_inference_time_s and self.processing_rate > 0:
            # Continue task, processing is going but not completed
            self.processing_state = 1
        else:
            # No Processing
            self.processing_state = 0
        """

        
        self.last_harvested_energy_wh = self.pv_instant_w * update_delta_time_h
        self.total_produced_energy_wh += self.last_harvested_energy_wh
        self.last_energy_used_wh = (self.base_load_energy_w + (self.full_load_energy_w*self.processing_state)) * update_delta_time_h
        self.total_consumed_energy_wh += self.last_energy_used_wh
        self.total_processing_time_s += h2s(update_delta_time_h) * self.processing_state

        # Update day or night
        if not self.is_day and self.pv_instant_w > 0 and s2h(time_s-self.day_end_time_s) > 1: #1h of hysteresis
            self.is_day = True
            self.day_start_time_s = time_s
            #print(f"Day started at {self.day_start_time_s} s")
        elif self.is_day and self.pv_instant_w == 0 and s2h(time_s-self.day_start_time_s) > 1: #1h of hysteresis
            self.is_day = False
            self.day_end_time_s = time_s
            self.plot_added = False

        if self.is_day:
            self.local_energy_counter_wh+=self.last_harvested_energy_wh
            if self.local_energy_counter_wh > self.battery_max_capacity_wh + (self.base_load_energy_w + self.full_load_energy_w)*s2h(time_s-self.day_start_time_s) and not self.is_maximum_reached:
                self.maximum_battery_time_s = time_s
                self.is_maximum_reached = True
        else:
            self.local_energy_counter_wh = 0
            self.is_maximum_reached = False
            if not self.plot_added:
                a = (self.day_start_time_s, self.day_end_time_s, self.day_end_time_s) if self.maximum_battery_time_s == 0 else (self.day_start_time_s,self.maximum_battery_time_s,self.day_end_time_s)
                #a = (self.day_start_time_s, self.maximum_battery_time_s)
                if a != (0,0,0):
                    self.maximum_rates.append(a)
                self.plot_added = True

        # Update battery level
        self.battery_current_capacity_wh += self.last_harvested_energy_wh - self.last_energy_used_wh
        self.battery_current_capacity_wh = max(
            0, self.battery_current_capacity_wh)
        
        # Accumulate wasted energy if battery is full
        if self.battery_current_capacity_wh > self.battery_max_capacity_wh:
            self.last_wasted_energy_wh = (self.battery_current_capacity_wh - self.battery_max_capacity_wh)/1000
            self.total_wasted_energy_wh += self.last_wasted_energy_wh
            self.battery_current_capacity_wh = self.battery_max_capacity_wh
        else:
            self.last_wasted_energy_wh = 0

        # print(f"T: {time_s} s | Next: {self.next_inference_time_s} | Pr: {self.processing_rate} | Battery: {self.battery_current_capacity_nah} nAh | Energy consumed by baseload: {energy_used_by_baseload_na} nAh | Energy used by task {energy_used_by_task_na} nAh | Energy harvested: {energy_produced_na} nAh")

        # Update statistics
        self.energy_consumption_w.append(self.base_load_energy_w+self.full_load_energy_w*self.processing_state)
        self.energy_harvested_w.append(self.pv_instant_w)
        self.battery_levels_percentage.append(
            self.battery_current_capacity_wh/self.battery_max_capacity_wh)
        self.processing_rates.append(self.processing_rate)
        self.times.append(time_s)

    def is_sunrise(self, time_s: int) -> bool:
        return self.is_day and time_s == self.day_start_time_s
    
    def is_sunset(self, time_s: int) -> bool:
        return not self.is_day and time_s == self.day_end_time_s
    
    def get_energy_consumption_w(self) -> float:
        return self.last_energy_used_wh

    def get_battery_percentage(self) -> float:
        return self.battery_current_capacity_wh/self.battery_max_capacity_wh

    def get_energy_efficiency(self) -> float:
        return (self.total_consumed_energy_wh/self.total_produced_energy_wh) if self.total_produced_energy_wh > 0 else 0

    def show_plot(self, show: bool = False, save: bool = False, filename: str = "test.png", additional_data:list = None, start_from:int = 0):
        fig, axs = plt.subplots(5 if additional_data else 4)
        axs[0].set_title("Energy consumption (w)")
        axs[0].plot(self.times, self.energy_consumption_w,
                    label="Energy consumption")
        axs[1].set_title("Processing rate (%)")
        axs[1].plot(self.times, self.processing_rates,
                    label="Processing rate")
        axs[2].set_title("Energy harvested (w)")
        axs[2].plot(self.times, self.energy_harvested_w,
                    label="Energy harvested")
        axs[3].set_title("Battery level (%)")
        axs[3].plot(self.times, self.battery_levels_percentage,
                    label="Battery level")
        axs[2].vlines(self.maximum_rates,ymin=0,ymax=40, color="g")
        axs[3].vlines(self.maximum_rates,ymin=0,ymax=1, color="g")
        if additional_data:
            axs[4].set_title("Additional data")
            axs[4].plot(self.times, additional_data,
                    label="Additional data")
        plt.legend()
        for i in range(len(axs)):
            axs[i].set_xlim(left=start_from)
        if save:
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()

    def reset(self, time_s:int, battery_percentage: float = 0.5):
        self.battery_current_capacity_wh = self.battery_max_capacity_wh*battery_percentage
        self.last_update_s = time_s
        self.processing_rate = 0
        self.set_pv_production_current_w(0)
        self.total_processing_time_s = 0

        # Clear arrays
        self.energy_consumption_w.clear()
        self.energy_harvested_w.clear()
        self.battery_levels_percentage.clear()
        self.processing_rates.clear()
        self.times.clear()

        #print("[DEVICE]Reset | Batt:",self.get_battery_percentage())

    def is_inference_finished(self, time_s:int) -> bool:
        return self.next_inference_time_s == 0
    

