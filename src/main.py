from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from lib.env_day import EnvDay, StateContent
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import json
import torch
import numpy as np
import random
from stable_baselines3.common.monitor import Monitor
from functools import partial

'''
Stato solo batteria, epochs 5, 3110 immagini, Update model on done or terminated

'''
parser = argparse.ArgumentParser()
parser.add_argument("--steps",default=10, required=False, type=int)
parser.add_argument("--update_steps",default=512, required=False, type=int)
parser.add_argument("--epochs",default=5, required=False, type=int)
parser.add_argument("--alg", default="ppo", choices=["ppo","a2c","dqn"])
parser.add_argument("--use_solar", default=False, action="store_true")
parser.add_argument("--use_month", default=False, action="store_true")
parser.add_argument("--use_hour", default=False, action="store_true")
parser.add_argument("--use_day", default=False, action="store_true")
parser.add_argument("--use_next_day", default=False, action="store_true")
parser.add_argument("--use_forecast", default=False, action="store_true")
parser.add_argument("--use_humidity", default=False, action="store_true")
parser.add_argument("--use_cloud", default=False, action="store_true")
parser.add_argument("--use_pressure", default=False, action="store_true")
parser.add_argument("--run_folder", default="../runs", type=str)
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--val",default=False, action="store_true")
parser.add_argument("--model",default=None, type=str)
parser.add_argument("--lr",type=float, default=0.0003)
parser.add_argument("--n_env", default=1, type=int, required=False)
parser.add_argument("--term_days", default=7, type=int, required=False)
args = parser.parse_args()
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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

state_content = 0
if args.use_solar:
    state_content ^= StateContent.SOLAR
if args.use_month:
    state_content ^= StateContent.MONTH
if args.use_hour:
    state_content ^= StateContent.HOUR
if args.use_day:
    state_content ^= StateContent.DAY
if args.use_next_day:
    state_content ^= StateContent.NEXT_DAY
if args.use_forecast:
    state_content ^= StateContent.SUN_PREDICTION
if args.use_pressure:
    state_content ^= StateContent.PRESSURE
if args.use_cloud:
    state_content ^= StateContent.CLOUD
if args.use_humidity:
    state_content ^= StateContent.HUMIDITY

class MyEvalCallBack(BaseCallback):
    def __init__(self, env : EnvDay, best_model_path:str, eval_freq:int = 0, verbose = 0):
        super().__init__(verbose)
        self.env = env
        self.best_images = 0
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            done = False
            obs, _ = self.env.reset()
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    if self.env.processed_images > self.best_images:
                        self.best_images = self.env.processed_images
                        self.model.save(self.best_model_path)
                        print(f"New best model with {self.env.processed_images} processed images")
        return True

test_env = EnvDay(csv_solar_data_path="../solcast2025.csv", step_s=5*60, state_content=state_content, random_reset=False, terminated_days=args.term_days)
#eval_env = Monitor(test_env)
eval_callback = MyEvalCallBack(test_env, best_model_path=os.path.join(args.run_folder,'best_model'), eval_freq=100000)

device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
#env = EnvDay(csv_solar_data_path="../solcast2024.csv", step_s=5*60, state_content=state_content)
vec_env = make_vec_env(lambda: EnvDay(csv_solar_data_path="../solcast2024.csv", step_s=5*60, state_content=state_content, terminated_days=args.term_days), n_envs=args.n_env)
alg_entry = None
if args.alg == "ppo":
    alg_entry = partial(PPO,
                        n_epochs=args.epochs,
                        n_steps = args.update_steps,
)
elif args.alg == "a2c":
    alg_entry = partial(A2C,
                        n_steps = args.update_steps,
)
elif args.alg == "dqn":
    alg_entry = partial(DQN,
                        train_freq=args.update_steps)
model = alg_entry(
    policy="MlpPolicy",
    env = vec_env,
    device=device,
    learning_rate=args.lr,
    verbose=2,
    tensorboard_log=os.path.join(args.run_folder,"tensorboard")
)

if not args.val:
    model.learn(total_timesteps=args.steps, progress_bar=True, callback=eval_callback)
    if args.run_folder is not None:
        model_path = os.path.join(args.run_folder,"last.pth")
        print("Model saved in",model_path)
        model.save(model_path)

    print("Trainign completed")

path_to_load = None
if args.val and args.model is not None:
    print("Loading model",args.model)
    path_to_load = args.model
else:
    best_model_path = os.path.join(args.run_folder, 'best_model')
    print("Loading model from",best_model_path)
    path_to_load = best_model_path

if args.alg == "ppo":
    model = PPO.load(path_to_load)
elif args.alg == "a2c":
    model = A2C.load(path_to_load)
elif args.alg == "dqn":
    model = DQN.load(path_to_load)
    
def save_plot(field:dict, path:str):
    # --- Plotting ---
    # sunrises = list(filter(lambda x: x[1] in sunrises , enumerate(times)))
    # sunrises = list(map(lambda x:x[0], sunrises))

    plt.figure(figsize=(12, 5))
    # plt.plot(battery_levels, label="Battery Level")
    # plt.plot(recharges, label="Recharge Amount")
    # plt.plot(consumptions, label="Energy Consumed")
    for f_name in fields.keys():
        plt.plot(fields[f_name], label=f_name)
    #plt.plot(forecast, label="Energy forecast")
    #plt.vlines(sunrises,ymin=0,ymax=1,colors="violet")
    # start_time = min(times).strftime("%H:%M %d/%m/%y")
    # end_time = max(times).strftime("%H:%M %d/%m/%y")
    # plt.title(f"Simulation from {start_time} untill {end_time}")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)

processed_images = []
#for w in range(0,120,1):
test_obs, info = test_env.reset(options={"norandom":True})
battery_levels, recharges, consumptions, times, forecast = [], [], [], [], []
battery_levels.append(test_obs[0])
done = False
rewards = []
is_plotted = False
steps = 0
plot_time = 60*60*24*31
unprocessed_images = 0
fields = {}
for f in info["fields"]:
    fields[f]=[]
fields["Energy"]=[]
while True:
    action,_ = model.predict(test_obs, deterministic=True)
    test_obs, reward, done, terminated, info = test_env.step(action)
    rewards.append(reward)

    if steps % 1000 == 0:
        print("Processing day",test_env.get_uptime_s()//(60*60*24),"of 121 | Processed images",test_env.processed_images, end="\r")
    if args.run_folder != "../runs" and (test_env.get_uptime_s() >= plot_time or done):
        plot_time += 60*60*24*31
        month = int(test_env.get_uptime_s() // (60*60*24*31))
        print("Plotting", month)
        # Limit da for 1 week
        steps_for_1_week = (60*60*24*7)//300
        for f in fields.keys():
            fields[f]=fields[f][:steps_for_1_week]
        image_folder = os.path.join(args.run_folder,"images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image{month}.jpg")

        save_plot(fields, img_file)
        for f in fields.keys():
            fields[f].clear()
    if done:
        print()
        print("BATTERY DEPLETED!")
        break
    
    # battery_levels.append(test_obs[0])
    # recharges.append(info["recharge"])
    # consumptions.append(info["cons"])
    # times.append(info["time"])
    # if args.use_forecast:
    #     forecast.append(test_obs[-1])
    for value,name in list(zip(test_obs, fields.keys())):
        fields[name].append(value)
    fields["Energy"].append(info["cons"])
    steps+=1

print("Ttime reached", test_env.get_human_uptime(), test_env.get_uptime_s())
print("Final reward", sum(rewards))
print("Processed images", test_env.processed_images)
print("Unprocessed images",unprocessed_images)
processed_images.append(test_env.processed_images)

print(processed_images)