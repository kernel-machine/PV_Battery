import unittest
from lib.device import Device
from lib.utils import s2h


class TestDevice(unittest.TestCase):
    def test_battery_percentage(self):
        device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=3300,
            battery_nominal_voltage_v=3.7,
            task_duration_s=1
        )
        device.reset(0,battery_percentage=0.3)
        self.assertEqual(0.3, device.get_battery_percentage())

    def test_processed_images(self):
        device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=3300,
            battery_nominal_voltage_v=3.7,
            task_duration_s=1
        )
        device.set_pv_production_current_w(20)
        device.set_processing_rate(1)
        device.update(0)
        device.update(5)  # 5 processate
        self.assertEqual(device.total_processed_images, 5)
        device.update(7)  # 7 processate
        device.update(5)  # Skippato
        self.assertEqual(device.total_processed_images, 7)
        device.set_processing_rate(0)  # t=10
        device.update(10)  # proccessing_rate = 0 viene applicato a t=10
        device.update(12)  # 7 Processate
        self.assertEqual(device.total_processed_images, 10)

    def test_energy_consumption(self):
        device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=1000,
            battery_nominal_voltage_v=3.7,
            device_nominal_voltage_v=5,
            task_duration_s=1
        )
        device.reset(0,battery_percentage=0.5)  # 1000 mah
        original_batt_wh = device.battery_current_capacity_wh
        device.set_pv_production_current_w(0)
        device.set_processing_rate(1)  # Applied at t=1
        device.update(1)
        batt_minus_1_img = original_batt_wh - \
            s2h(1)*(device.base_load_energy_w+device.full_load_energy_w)
        self.assertEqual(device.battery_current_capacity_wh,
                         batt_minus_1_img, "Battery consumption after 1 second")

        device.update(10)
        original_minus_10_img = original_batt_wh - \
            s2h(10)*(device.base_load_energy_w + device.full_load_energy_w)
        self.assertAlmostEqual(device.battery_current_capacity_wh,
                               original_minus_10_img, 4, "Battery consumption after 10 seconds")

        device.set_processing_rate(0)
        # Processing rate setted to 0 at t=11, 10 images processed since processing start at t=1
        device.update(11)
        self.assertEqual(device.total_processed_images, 10)
        battery_processing_end = device.battery_current_capacity_wh

        device.update(80)
        batt = battery_processing_end - s2h(80-11)*device.base_load_energy_w
        self.assertAlmostEqual(device.battery_current_capacity_wh,
                               batt, "Battery consumption with an idle state")

    def test_solar_chage(self):
        device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=1000,
            battery_nominal_voltage_v=3.7,
            device_nominal_voltage_v=5,
            task_duration_s=1
        )
        device.reset(0,battery_percentage=0.5)
        device.set_pv_production_current_w(10)  # PV panel set to 10w
        device.update(0)  # PV production takes effect
        start_battery_wh = device.battery_current_capacity_wh
        recarghed_capacity_wh = s2h(10)*10
        idle_capacity_consumption = s2h(10)*device.base_load_energy_w
        device.update(10)  # After 10 seconds
        self.assertAlmostEqual(device.battery_current_capacity_wh, start_battery_wh +
                               recarghed_capacity_wh-idle_capacity_consumption, 4, "PV Solar charging")


    def test_energy_consumption(self):
        device = Device(
            base_load_energy_ma=50,
            full_load_energy_ma=1000,
            battery_max_capacity_mah=1000,
            battery_nominal_voltage_v=3.7,
            device_nominal_voltage_v=5,
            task_duration_s=1
        )
        device.reset(0,battery_percentage=0.5)
        device.set_pv_production_current_w(0)
        device.update(0)
        device.update(1)
        self.assertEqual(device.get_energy_consumption_w(),0.050*5*s2h(1),"Checking idle consumption") #w=v*i
        device.set_processing_rate(1)
        device.update(2)
        device.update(3)
        self.assertEqual(device.get_energy_consumption_w(),1.050*5*s2h(1),"Checking full power consumption") #w=v*i
        

if __name__ == "__main__":
    unittest.main()
