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
import shutil
import math
from sb3_contrib import RecurrentPPO
'''
Stato solo batteria, epochs 5, 3110 immagini, Update model on done or terminated

'''
parser = argparse.ArgumentParser()
parser.add_argument("--steps",default=10, required=False, type=int)
parser.add_argument("--update_steps",default=512, required=False, type=int)
parser.add_argument("--epochs",default=5, required=False, type=int)
parser.add_argument("--alg", default="ppo", choices=["ppo","a2c","dqn","rec_ppo"])
parser.add_argument("--use_solar", default=False, action="store_true")
parser.add_argument("--use_month", default=False, action="store_true")
parser.add_argument("--use_hour", default=False, action="store_true")
parser.add_argument("--use_minute", default=False, action="store_true")
parser.add_argument("--use_day", default=False, action="store_true")
parser.add_argument("--use_next_day", default=False, action="store_true")
parser.add_argument("--use_real_forecast", default=False, action="store_true")
parser.add_argument("--use_estimate_forecast", default=False, action="store_true")
parser.add_argument("--use_estimate_single_forecast", default=False, action="store_true")
parser.add_argument("--use_humidity", default=False, action="store_true")
parser.add_argument("--use_cloud", default=False, action="store_true")
parser.add_argument("--use_pressure", default=False, action="store_true")
parser.add_argument("--run_folder", default="../runs", type=str)
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--val",default=False, action="store_true")
parser.add_argument("--model",default=None, type=str)
parser.add_argument("--lr",type=float, default=0.0003)
parser.add_argument("--lr_scheduler",default=None, choices=["exp","cos","lin","lin_half"])
parser.add_argument("--n_env", default=1, type=int, required=False)
parser.add_argument("--term_days", default=7, type=int, required=False)
parser.add_argument("--forecast_minutes",type=int, default=60)
parser.add_argument("--prediction_accuracy", default=1.0, type=float)
parser.add_argument("--choose_forecast",default=False, action="store_true")
parser.add_argument("--test_year",type=int, default=2025)
parser.add_argument("--test_freq",type=int, default=0)
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
if args.use_real_forecast:
    state_content ^= StateContent.SUN_REAL_PREDICTION
if args.use_estimate_forecast:
    state_content ^= StateContent.SUN_ESTIMATE_PREDICTION
if args.use_estimate_single_forecast:
    state_content ^= StateContent.SUN_ESTIMATE_SINGLE_PREDICTION
if args.use_pressure:
    state_content ^= StateContent.PRESSURE
if args.use_cloud:
    state_content ^= StateContent.CLOUD
if args.use_humidity:
    state_content ^= StateContent.HUMIDITY
if args.use_minute:
    state_content ^= StateContent.MINUTE

class MyEvalCallBack(BaseCallback):
    def __init__(self, env : EnvDay, best_model_path:str, eval_freq:int = 0, verbose = 0):
        super().__init__(verbose)
        self.env = env
        self.best_images = 0
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq
        self.next_call = eval_freq

    def _on_step(self):
        #if self.n_calls % 10 == 0: print("Checking callback",self.eval_freq > 0 , self.n_calls > self.next_call, self.n_calls , self.next_call)
        if self.eval_freq > 0 and self.n_calls > self.next_call:
            print("Triggered callback",self.n_calls, self.next_call)
            self.next_call+=self.eval_freq
            obs, _ = self.env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:# or truncated:
                    if self.env.processed_images > self.best_images:
                        self.best_images = self.env.processed_images
                        self.model.save(self.best_model_path)
                        self.logger.record("env_images",self.env.processed_images)
                        print(f"New best model with {self.env.processed_images} processed images")
                    break
        return True

test_env = EnvDay(csv_solar_data_path=f"../solcast{args.test_year}.csv", step_s=5*60, seed=SEED, state_content=state_content, random_reset=False, terminated_days=args.term_days, forecast_time_m=args.forecast_minutes, prediction_accuracy=args.prediction_accuracy, choose_forecast=args.choose_forecast)
eval_callback = MyEvalCallBack(test_env, best_model_path=os.path.join(args.run_folder,'best_model'), eval_freq=args.test_freq)
print("test env created")

device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
#env = EnvDay(csv_solar_data_path="../solcast2024.csv", step_s=5*60, state_content=state_content)
vec_env = make_vec_env(lambda: EnvDay(csv_solar_data_path="../solcast2024.csv", step_s=5*60, seed=SEED, state_content=state_content, random_reset=True, terminated_days=args.term_days, forecast_time_m=args.forecast_minutes, prediction_accuracy=args.prediction_accuracy, choose_forecast=args.choose_forecast), n_envs=args.n_env, )
print("vec env created")

def linear_schedule(initial_value, start_from:float = 0):
    def func(progress_remaining: float) -> float:
        if progress_remaining < 1-start_from:
            return initial_value * (progress_remaining/(1-start_from))
        else:
            return initial_value
    return func   

def exp_schedule(initial_value, decay_rate=5):
    def func(progress_remaining: float) -> float:
        return initial_value * math.exp(-decay_rate * (1 - progress_remaining))
    return func

def cosine_schedule(initial_value):
    def func(progress_remaining: float) -> float:
        return initial_value * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
    return func

if args.lr_scheduler is None:
    lr = args.lr
    print(f"Fixed LR: {lr}")
elif args.lr_scheduler == "exp":
    lr = exp_schedule(args.lr)
    print("Exponential LR descrese")
elif args.lr_scheduler == "cos":   
    lr = cosine_schedule(args.lr)
    print("Cosine LR descrese")
elif args.lr_scheduler == "lin":   
    lr = linear_schedule(args.lr)
    print("Linear LR descrese")
elif args.lr_scheduler == "lin_half":   
    lr = linear_schedule(args.lr, start_from=0.5)
    print("Linear LR from half descrese")

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
elif args.alg == "rec_ppo":
    alg_entry = partial(RecurrentPPO,
                        n_epochs=args.epochs,
                        n_steps = args.update_steps,
                        )
    
policy = "MlpLstmPolicy" if args.alg == "rec_ppo" else "MlpPolicy"
model = alg_entry(
    policy=policy,
    env = vec_env,
    device=device,
    learning_rate=lr,
    verbose=2,
    tensorboard_log="../tensorboard_logs"#os.path.join(args.run_folder,"tensorboard")
)
print("Model created")

if not args.val:
    try:
        model.learn(total_timesteps=args.steps, progress_bar=True, callback=eval_callback)
        if args.run_folder is not None:
            model_path = os.path.join(args.run_folder,"last.pth")
            print("Model saved in",model_path)
            model.save(model_path)

        print("Trainign completed")
    except:
        print("Cleaning run folder",args.run_folder)
        shutil.rmtree(args.run_folder)

path_to_load = None
best_model_path = os.path.join(args.run_folder, 'best_model')
last_model_path = os.path.join(args.run_folder, "last.pth")
if args.val and args.model is not None:
    print("Loading model",args.model)
    path_to_load = args.model
elif os.path.exists(best_model_path):
    print("Loading model from",best_model_path)
    path_to_load = best_model_path
else:
    print("Loading model from",last_model_path)
    path_to_load = last_model_path

if args.alg == "ppo":
    model = PPO.load(path_to_load)
elif args.alg == "a2c":
    model = A2C.load(path_to_load)
elif args.alg == "dqn":
    model = DQN.load(path_to_load)
elif args.alg == "rec_ppo":
    model = RecurrentPPO.load(path_to_load)
    
def save_plot(field:dict, path:str, views:list = None):
    # --- Plotting ---
    # sunrises = list(filter(lambda x: x[1] in sunrises , enumerate(times)))
    # sunrises = list(map(lambda x:x[0], sunrises))

    plt.figure(figsize=(12, 5))
    # plt.plot(battery_levels, label="Battery Level")
    # plt.plot(recharges, label="Recharge Amount")
    # plt.plot(consumptions, label="Energy Consumed")
    for f_name in fields.keys():
        if views is not None and f_name in views:
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

if test_env.choose_forecast:
    fields["start_time"]=[]
    fields["window_time_s"]=[]

# Remove fields
needed_views = ["Battery","Solar","Hour","Energy"]

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

        save_plot(fields, img_file, needed_views)
        for f in fields.keys():
            fields[f].clear()
    if done or test_env.get_uptime_s() >= 120*24*60*60:
        print()
        print("BATTERY DEPLETED!")
        break
    
    for value,name in list(zip(test_obs, fields.keys())):
        fields[name].append(value)
    if "Energy" in fields.keys(): fields["Energy"].append(info["cons"])
    if test_env.choose_forecast:
        fields["start_time"].append(info["forecast"][0])
        fields["window_time_s"].append(info["forecast"][1])
    steps+=1

print("Ttime reached", test_env.get_human_uptime(), test_env.get_uptime_s())
print("Final reward", sum(rewards))
print("Processed images", test_env.processed_images)
print("Unprocessed images",unprocessed_images)
processed_images.append(test_env.processed_images)

print(processed_images)