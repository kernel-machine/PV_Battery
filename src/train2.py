from lib.rl.ppo2 import PPO
from lib.env import NodeEnv
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import json
import torch
'''
Stato solo batteria, epochs 5, 3110 immagini, Update model on done or terminated

'''
parser = argparse.ArgumentParser()
parser.add_argument("--steps",default=128, required=False, type=int)
parser.add_argument("--update_steps",default=256, required=False, type=int)
parser.add_argument("--epochs",default=10, required=False, type=int)
parser.add_argument("--use_solar", default=False, action="store_true")
parser.add_argument("--use_month", default=False, action="store_true")
parser.add_argument("--use_hour", default=False, action="store_true")
parser.add_argument("--use_day", default=False, action="store_true")
parser.add_argument("--run_folder", default="../runs", type=str)
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--val",default=False, action="store_true")
parser.add_argument("--model",default=None, type=str)
args = parser.parse_args()
SEED = 42

if not args.val and args.run_folder == "../runs":
    folders = os.listdir("../runs")
    folders = list(filter(lambda x:x.isdigit(), folders))
    last_id = max(map(lambda x:int(x),folders))
    new_id = last_id+1
    args.run_folder = os.path.join(args.run_folder,str(new_id))

if args.run_folder is not None:
    if os.path.exists(args.run_folder) and not args.val:
        print("Run folder already exists!")
        exit(0)
    else:
        os.makedirs(args.run_folder, exist_ok=True)
        print("Files were saved on",args.run_folder)

if args.run_folder is not None and not args.val:
    file = os.path.join(args.run_folder,"args.json")
    with open(file, 'w', encoding='utf-8') as f:
        j = json.dumps(args.__dict__, ensure_ascii=False, indent=4)
        f.write(j)


device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
env = NodeEnv(csv_solar_data_path="../solcast2024.csv", step_s=5*60, use_solar=args.use_solar, use_month=args.use_month, use_hour=args.use_hour, use_day=args.use_day)
ppo = PPO(env, seed=SEED, n_steps=args.update_steps, gae_lambda=0.99, batch_size=32, max_terminated=0, learning_rate=1e-3, epochs=args.epochs, run_folder=args.run_folder, device=device)
if not args.val:
    ppo.learn(512*args.steps)
    if args.run_folder is not None:
        model_path = os.path.join(args.run_folder,"last.pth")
        print("Model saved in",model_path)
        ppo.save(model_path)

    print("Trainign completed")

if args.val and args.model is not None:
    print("Loading model",args.model)
    ppo.load(args.model)
    
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

test_env = NodeEnv(csv_solar_data_path="../solcast2025.csv", step_s=5*60, use_solar=args.use_solar, use_month=args.use_month, use_hour=args.use_hour, use_day=args.use_day)
test_obs,_ = test_env.reset(options={"norandom":True})
battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
battery_levels.append(test_obs[0])
done = False
rewards = []
is_plotted = False
steps = 0
plot_time = 60*60*24*31
unprocessed_images = 0

while True:
    action = ppo.select_action(test_obs)
    test_obs, reward, done, terminated, info = test_env.step(action)
    rewards.append(reward)

    # if info["sunrise"]:
    #     #sunrises.append(info["time"])
    #     image_energy_j = test_env.device_full_energy_w*test_env.step_size_s
    #     idle_energy_j = test_env.device_idle_energy_w*test_env.step_size_s
    #     processable_images = test_env.battery_curr_j/image_energy_j
    #     energy_used_by_idle_j = idle_energy_j*processable_images
    #     processable_images += energy_used_by_idle_j/image_energy_j
    #     unprocessed_images += int(processable_images)

    if steps % 1000 == 0:
        print("Processing day",test_env.get_uptime_s()//(60*60*24),"of 121 | Processed images",test_env.processed_images, end="\r")
    if args.run_folder != "../runs" and (test_env.get_uptime_s() > plot_time or done):
        plot_time += 60*60*24*31
        month = test_env.get_uptime_s() // (60*60*24*31)
        print("Plotting", month)
        # Limit da for 1 week
        steps_for_1_week = (60*60*24*7)//300
        battery_levels = battery_levels[:steps_for_1_week]
        recharges = recharges[:steps_for_1_week]
        consumptions = consumptions[:steps_for_1_week]
        times = times[:steps_for_1_week]
        #sunrises = list(filter(lambda x:x<=max(times), sunrises))
        # Creating folders
        image_folder = os.path.join(args.run_folder,"images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image{month}.jpg")

        save_plot(battery_levels, recharges, consumptions, times, sunrises, img_file)
        battery_levels, recharges, consumptions, times, sunrises = [], [], [], [], []
    if done:
        print()
        print("BATTERY DEPLETED!")
        break
    
    battery_levels.append(test_obs[0])
    recharges.append(info["recharge"])
    consumptions.append(info["cons"])
    times.append(info["time"])
    steps+=1
print("Ttime reached", test_env.get_human_uptime())
print("Final reward", sum(rewards))
print("Processed images", test_env.processed_images)
print("Unprocessed images",unprocessed_images)

