from collections.abc import Callable
import matplotlib.pyplot as plt


def ms2h(ms: int) -> float:
    return ms/3600000.0


def h2ms(h: float) -> int:
    return h*3600000.0


def s2h(s: float) -> float:
    return s/3600.0


class Device:

    def __init__(self,
                 base_load_energy_ma: int,
                 full_load_energy_ma: int,
                 battery_max_capacity_mah: int,
                 task_duration_s: int,
                 init_battery_percentage: float = 0.5,
                 processing_rate_force_update_after_s: int = 0,):

        # Device specific parameters
        self.base_load_energy_na = base_load_energy_ma*1000
        self.full_load_energy_na = full_load_energy_ma*1000
        self.battery_max_capacity_nah = battery_max_capacity_mah*1000
        self.battery_current_capacity_nah = int(
            self.battery_max_capacity_nah*init_battery_percentage)  # Battery at 50%

        # PV specifications
        self.pv_instant_production_na = 0

        # Algoritm parameters
        self.processing_rate: float = 0
        self.next_inference_time_s = 0
        self.last_update_s = 0  # Used for energy estimation
        self.task_start_time_s = 0
        self.task_duration_s = task_duration_s
        self.processing_rate_force_update_after_s = processing_rate_force_update_after_s

        # Store statistics
        self.times = []
        self.battery_levels_percentage = []
        self.energy_consumption_ma = []
        self.energy_harvested_ma = []
        self.processing_rates = []
        self.total_processed_images = 0
        self.total_consumed_energy_mah = 0
        self.total_produced_energy_mah = 0
        self.total_wasted_energy_mah = 0

        self.last_wasted_energy_mah = 0
        self.last_harvested_energy_nah = 0
        self.last_energy_used_nah = 0

    # Update the instant production of the PV
    def set_pv_production_current_na(self, pv_current_na: int):
        self.pv_instant_production_na = pv_current_na

    def set_pv_production_current_ma(self, pv_current_ma: int):
        self.set_pv_production_current_na(pv_current_ma*1000)

    def get_pv_production_normalized(self) -> float:
        return (self.pv_instant_production_na / 1000000)/2.31

    def set_processing_rate(self, processing_rate: float):
        self.processing_rate = max(0.0, min(processing_rate, 1.0))

    def update(self, time_s: float):
        # print("Check")
        if self.last_update_s == 0:
            self.last_update_s = time_s
        update_delta_time_h = s2h(time_s-self.last_update_s)
        self.last_update_s = time_s
        """ or self.is_processing_rate_updated"""
        if (time_s > self.next_inference_time_s) and self.processing_rate > 0:
            # Start a new task
            self.task_start_time_s = time_s
            wait_time_s = (self.task_duration_s) * \
                (1-self.processing_rate)/self.processing_rate

            if self.processing_rate_force_update_after_s > 0:  # To avoid infinite wait time
                self.next_inference_time_s = time_s + \
                    min(wait_time_s, self.processing_rate_force_update_after_s)
            else:
                self.next_inference_time_s = time_s + wait_time_s

            processing_state = 1
            self.total_processed_images += 1
        elif time_s < self.task_start_time_s + self.task_duration_s and self.processing_rate > 0:
            # Continue task
            processing_state = 1
        else:
            # No Processing
            processing_state = 0

        self.last_harvested_energy_nah = self.pv_instant_production_na * update_delta_time_h
        self.total_produced_energy_mah += (self.last_harvested_energy_nah/1000)
        self.last_energy_used_nah = (self.base_load_energy_na + (self.full_load_energy_na*processing_state)) * update_delta_time_h
        self.total_consumed_energy_mah += (self.last_energy_used_nah/1000)

        # Update battery level
        self.battery_current_capacity_nah += self.last_harvested_energy_nah - self.last_energy_used_nah
        self.battery_current_capacity_nah = max(
            0, self.battery_current_capacity_nah)
        
        # Accumulate wasted energy if battery is full
        if self.battery_current_capacity_nah > self.battery_max_capacity_nah:
            self.last_wasted_energy_mah = (self.battery_current_capacity_nah - self.battery_max_capacity_nah)/1000
            self.total_wasted_energy_mah += self.last_wasted_energy_mah
            self.battery_current_capacity_nah = self.battery_max_capacity_nah
        else:
            self.last_wasted_energy_mah = 0

        # print(f"T: {time_s} s | Next: {self.next_inference_time_s} | Pr: {self.processing_rate} | Battery: {self.battery_current_capacity_nah} nAh | Energy consumed by baseload: {energy_used_by_baseload_na} nAh | Energy used by task {energy_used_by_task_na} nAh | Energy harvested: {energy_produced_na} nAh")

        # Update statistics
        self.energy_consumption_ma.append(
            (self.base_load_energy_na+self.full_load_energy_na*processing_state)/1000)
        self.energy_harvested_ma.append(self.pv_instant_production_na/1000)
        self.battery_levels_percentage.append(
            self.battery_current_capacity_nah/self.battery_max_capacity_nah)
        self.processing_rates.append(self.processing_rate)
        self.times.append(time_s)

    def get_energy_consumption_ma(self) -> float:
        return (self.base_load_energy_na + self.full_load_energy_na * self.processing_rate)/1000

    def get_energy_consumption_a(self) -> float:
        return self.get_energy_consumption_ma()/1000

    def get_battery_percentage(self) -> float:
        return self.battery_current_capacity_nah/self.battery_max_capacity_nah
    
    def get_energy_efficiency(self) -> float:
        return (self.total_consumed_energy_mah/self.total_produced_energy_mah) if self.total_produced_energy_mah > 0 else 0

    def show_plot(self, show: bool = False, save: bool = False, filename: str = "test.png"):
        fig, axs = plt.subplots(4)
        axs[0].set_title("Energy consumption (mA)")
        axs[0].plot(self.times, self.energy_consumption_ma,
                    label="Energy consumption")
        axs[1].set_title("Processing rate (%)")
        axs[1].plot(self.times, self.processing_rates,
                    label="Processing rate")
        axs[2].set_title("Energy harvested (mA)")
        axs[2].plot(self.times, self.energy_harvested_ma,
                    label="Energy harvested")
        axs[3].set_title("Battery level (%)")
        axs[3].plot(self.times, self.battery_levels_percentage,
                    label="Battery level")
        plt.legend()
        if save:
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()

    def reset(self, battery_percentage: float = 0.5):
        self.battery_current_capacity_nah = int(
            self.battery_max_capacity_nah*battery_percentage)
        self.last_update_s = 0
        self.processing_rate = 0
        self.set_pv_production_current_ma(0)
        self.energy_consumption_ma.clear()
        self.energy_harvested_ma.clear()
        self.battery_levels_percentage.clear()
        self.processing_rates.clear()
        self.times.clear()

    def is_inference_finished(self, time_s:int) -> bool:
        return self.next_inference_time_s == 0
    
if __name__ == "__main__":
    device = Device(100, 100, 1000, 10)
    device.set_processing_rate(0.1)
    device.set_pv_production_current_ma(2000)
    for i in range(0,60*60*24*5,60):
        device.update(i)
        print(f"Time: {i} s | Battery: {device.get_battery_percentage()}% | Processing rate: {device.processing_rate}")
    device.show_plot(show=False, save=True)