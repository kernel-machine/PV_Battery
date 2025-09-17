from lib.pid_controller import PIDController
from lib.env import NodeEnv
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--setpoint",default=0.35, type=float)
args = parser.parse_args()
controller = PIDController(kp=100, ki=0, dt=0)
battery_setpoint = args.setpoint  # Desired battery level (80%)

def save_plot(battery_levels: list, recharges: list, consumptions:list, times:list, sunrises, path:str):
    # --- Plotting ---
    sunrises = list(filter(lambda x: x[1] in sunrises , enumerate(times)))
    sunrises = list(map(lambda x:x[0], sunrises))

    plt.figure(figsize=(12, 5))
    plt.plot(battery_levels, label="Battery Level")
    plt.plot(recharges, label="Recharge Amount")
    plt.plot(consumptions, label="Energy Consumed")
    plt.vlines(sunrises,ymin=0,ymax=1,colors="violet")
    start_time = min(times).strftime("%H:%M %d/%m/%y")
    end_time = max(times).strftime("%H:%M %d/%m/%y")
    plt.title(f"Simulation from {start_time} untill {end_time}")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)

env = NodeEnv(csv_solar_data_path="../solcast2025.csv", step_s=5*60)
next_obs,_ = env.reset(options={"norandom":True})
steps = 0
battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
plot_time = 60*60*24*31
unprocessed_images = 0
while True:
    error = battery_setpoint - next_obs[0] #env.device.get_battery_percentage()
    control_output = -controller.update(error)
    action = 1 if control_output > 0 else 0
    next_obs, reward, done, terminated, info = env.step(action)
    battery_levels.append(next_obs[0])
    recharges.append(info["recharge"])
    consumptions.append(info["cons"])
    times.append(info["time"])
    if info["sunrise"]:
        sunrises.append(info["time"])
        image_energy_j = env.device_full_energy_w*env.step_size_s
        idle_energy_j = env.device_idle_energy_w*env.step_size_s
        processable_images = env.battery_curr_j/image_energy_j
        energy_used_by_idle_j = idle_energy_j*processable_images
        processable_images += energy_used_by_idle_j/image_energy_j
        unprocessed_images += int(processable_images)
    if done or next_obs[0]<=0:
        print("State", next_obs)
        break
    if steps % 1000 == 0:
        print("Processing ",env.processed_images, unprocessed_images, end="\r")
    if env.get_uptime_s() > plot_time or done:
        plot_time += 60*60*24*31
        month = env.get_uptime_s() // (60*60*24*31)
        print("Plotting", month)
        steps_for_1_week = (60*60*24*7)//300
        # Limit data to 1 week
        battery_levels = battery_levels[:steps_for_1_week]
        recharges = recharges[:steps_for_1_week]
        consumptions = consumptions[:steps_for_1_week]
        times = times[:steps_for_1_week]
        sunrises = list(filter(lambda x:x<=max(times), sunrises))
        # Creating folders
        image_folder = os.path.join("../runs/pid","images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image{month}.jpg")

        save_plot(battery_levels, recharges, consumptions, times, sunrises, img_file)
        battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
    steps+=1
print("Running completed")
print("Uptime",env.get_human_uptime())
print("Ttime reached", env.get_human_uptime())
print("Processed images", env.processed_images)
print("Processed images with inf batt",env.harvested_energy_j/(env.device_full_energy_w*env.step_size_s))
print("Unprocessed images",unprocessed_images)