from lib.pid_controller import PIDController
from lib.env_day import EnvDay
import argparse
import matplotlib.pyplot as plt
import os



def save_plot(battery_levels: list, recharges: list, consumptions:list, times:list, sunrises, path:str):
    # --- Plotting ---
    sunrises = list(filter(lambda x: x[1] in sunrises , enumerate(times)))
    sunrises = list(map(lambda x:x[0], sunrises))

    plt.figure(figsize=(12, 5))
    plt.plot(battery_levels, label="Battery Level")
    plt.plot(recharges, label="Recharge Amount")
    plt.plot(consumptions, label="Energy Consumed")
    plt.vlines(sunrises,ymin=0,ymax=1,colors="violet")
    if len(times)>0:
        start_time = min(times).strftime("%H:%M %d/%m/%y")
        end_time = max(times).strftime("%H:%M %d/%m/%y")
        plt.title(f"Simulation from {start_time} untill {end_time}")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)

if __name__ == "__main__":
    env = EnvDay(csv_solar_data_path="../solcast2025.csv", step_s=5*60)
    next_obs,_ = env.reset(options={"norandom":True})
    steps = 0
    battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
    plot_time = 60*60*24*2
    total_unprocessed_images = 0
    action = 0
    sunrise_watchdogs = 0
    hit_sunrise = False
    hit_sunset = False


    peak_reached = False
    max_battery = 0
    max_battery_patience = 5
    while True:
        
        next_obs, reward, done, terminated, info = env.step(action)
        battery_levels.append(next_obs[0])
        recharges.append(info["recharge"])
        consumptions.append(info["cons"])
        times.append(info["time"])
        sunrise_watchdogs += env.step_size_s

        if not peak_reached:
            if next_obs[0] > max_battery:
                max_battery = next_obs[0]
            else:
                max_battery_patience -= 1
            if max_battery_patience <= 0:
                peak_reached = True
                sunrises.append(info["time"])
        # Wait for a sunset
        if env.solar.is_sunrise(env.time_s, env.step_size_s, 5) and not hit_sunrise:
            hit_sunrise = True
            sunrises.append(info["time"])
        # Wait for energy
        if hit_sunrise and env.solar.are_steps_with_at_least(env.time_s, step_size_s=env.step_size_s,steps=10,power_w=env.device_full_energy_w):
            # Start processing
            #sunrises.append(info["time"])
            image_energy_j = env.device_full_energy_w*env.step_size_s
            idle_energy_j = env.device_idle_energy_w*env.step_size_s
            # Quante immagini posso processare
            processable_images = env.battery_curr_j/image_energy_j
            # Recupero l'energia dell'idle
            energy_used_by_idle_j = idle_energy_j*int(processable_images)
            # Aggiungo le immagini processare con l'energia dell'idle
            processable_images += energy_used_by_idle_j/image_energy_j

            total_unprocessed_images += int(processable_images)
            env.battery_curr_j += energy_used_by_idle_j
            env.battery_curr_j -= int(processable_images)*image_energy_j  
            sunrise_watchdogs = 0
            action = 1
            hit_sunrise = False
            peak_reached = False
            max_battery = 0
            max_battery_patience = 5

        # Stop when is night and the battery is less than 50%
        if next_obs[0]<0.3 and peak_reached:
            action = 0

        if info["recharge"]==0 and next_obs[0]==1:
            print("Max reached during the night")
        

        
        if steps % 1000 == 0:
            print("Processing ",env.processed_images, total_unprocessed_images, end="\r")
        if env.get_uptime_s() > plot_time or done or next_obs[0]<=0:
            plot_time += 60*60*24*31
            month = env.get_uptime_s() // (60*60*24*31)
            print("Plotting", month)
            steps_for_1_week = (60*60*24*5)//300
            # Limit data to 1 week
            end_of_month = False
            if end_of_month:
                battery_levels = battery_levels[-steps_for_1_week:]
                recharges = recharges[-steps_for_1_week:]
                consumptions = consumptions[-steps_for_1_week:]
                times = times[-steps_for_1_week:]
            else:
                battery_levels = battery_levels[:steps_for_1_week]
                recharges = recharges[:steps_for_1_week]
                consumptions = consumptions[:steps_for_1_week]
                times = times[:steps_for_1_week]

            sunrises = list(filter(lambda x:x<=max(times), sunrises))
            # Creating folders
            image_folder = os.path.join("../runs/optimal","images")
            os.makedirs(image_folder, exist_ok=True)
            img_file = os.path.join(image_folder,f"image{month}.jpg")

            save_plot(battery_levels, recharges, consumptions, times, sunrises, img_file)
            battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
            break
        if sunrise_watchdogs > 60*60*32 and info["recharge"]>=0:
            pass
            #print("Sunrise missed at",env.time_s)
            #exit(-1)
        if done:# or next_obs[0]<=0 or env.battery_curr_j<=0:
            print("State", next_obs)
            break
        steps+=1
    print("Running completed")
    print("Uptime",env.get_human_uptime())
    print("Ttime reached", env.get_human_uptime())
    print("Processed images", env.processed_images)
    print("Unprocessed images",total_unprocessed_images)
    print("Total processed images", env.processed_images+total_unprocessed_images)