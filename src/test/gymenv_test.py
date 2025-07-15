import unittest
from lib.env import NodeEnv
from lib.utils import s2h


class GymEnvTest(unittest.TestCase):
    def test_env_step(self):
        env = NodeEnv(
            "../solcast2024.csv",
            step_s=60,
            stop_on_full_battery=False,
            discrete_action=True,
            discrete_state=False
        )
        # Time start at 300, no PV production
        env.reset(options={"norandom": True})

        pre_battery_wh = env.device.battery_current_capacity_wh
        env.step(0)  # After 60 seconds of idle state
        post_battery_wh = env.device.battery_current_capacity_wh
        self.assertAlmostEqual(post_battery_wh, pre_battery_wh-s2h(
            60)*env.device.base_load_energy_w, 4, "Check battery consumption after a idle step")

        pre_battery_wh = env.device.battery_current_capacity_wh
        env.step(1)  # After 60 seconds of full state
        post_battery_wh = env.device.battery_current_capacity_wh
        self.assertAlmostEqual(post_battery_wh, pre_battery_wh-s2h(60)*(env.device.base_load_energy_w +
                               env.device.full_load_energy_w), 4, "Check battery consumption after a full step")

        # Wait for solar energy
        while env.device.get_pv_production_normalized() <= 0:
            env.step(0)

        pre_battery_wh = env.device.battery_current_capacity_wh
        env.step(0)  # After 60 seconds of idle state
        post_battery_wh = env.device.battery_current_capacity_wh
        consumed_capacity_wh = s2h(60)*env.device.base_load_energy_w
        harvested_capacity_wh = s2h(60)*env.device.pv_instant_w
        self.assertGreater(harvested_capacity_wh, 0, "Check harvested energy is positive")
        expected_battery_wh = pre_battery_wh - consumed_capacity_wh + harvested_capacity_wh
        self.assertAlmostEqual(post_battery_wh, expected_battery_wh,
                               4, "Check battery consumption considering PV")
        
    def test_reward(self):
        env = NodeEnv(
            "../solcast2024.csv",
            step_s=60,
            stop_on_full_battery=False,
            discrete_action=True,
            discrete_state=False
        )
        env.reset(options={"norandom": True})

        #Charge battery at 100%
        while env.device.get_battery_percentage() < 1:
            env.step(0)
        self.assertEqual(env.device.get_battery_percentage(), 1)
        reward_bad_action = env.step(0)[1]
        reward_good_action = env.step(1)[1]
        self.assertGreater(reward_good_action, reward_bad_action, "Checking reward on full battery")