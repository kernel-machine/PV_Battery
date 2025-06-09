from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from torch.utils.tensorboard import SummaryWriter
from lib.env import NodeEnv
import random
import datetime
import argparse

args = argparse.ArgumentParser()
args.add_argument("--short_training", default=False, action="store_true", help="Set to have a shorter training")
args.add_argument("--n_env", type=int, default=4, help="Number of environment to train in parallel")
args.add_argument("--val", default=False, action="store_true", help="Validate the model without training")
args.add_argument("--discrete_action", default=False, action="store_true", help="Use discrete actions instead of continuous")
args.add_argument("--discrete_state", default=False, action="store_true", help="Use discrete state space instead of continuous")

args = args.parse_args()
SEED = 1234
STEP_S = 60*15 # Step size in seconds
random.seed(SEED)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

if args.discrete_state:
    env = NodeEnv("../solcast2024.csv", step_s=STEP_S, stop_on_full_battery=True, discrete_action=args.discrete_action, discrete_state=args.discrete_state)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=2,
    )
    days = 365
    total_steps = (days * 24 * 60 * 60 // STEP_S)  # Total steps for training
else:
    vec_env = make_vec_env(
        NodeEnv,
        n_envs=args.n_env,
        seed=SEED,
        env_kwargs={"csv_solar_data_path": "../solcast2024.csv", "step_s": STEP_S, "stop_on_full_battery": True, "discrete_state": args.discrete_state,"discrete_action": args.discrete_action},
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        #tensorboard_log=log_dir,
        verbose=2,
        device="cpu",
        n_steps=(24*60*60)//STEP_S, # 1 Day
        seed=SEED,
        batch_size=64,
        gamma=0.99,
        policy_kwargs=dict(log_std_init=-2),
        #learning_rate=1e-5,
        n_epochs=10,
    )
    days = 365
    total_steps = (days * 24 * 60 * 60 // STEP_S) * args.n_env  # Total steps for training

if args.short_training:
    total_steps /= 4
if args.val:
    model = PPO.load("ppo_node_env", env=vec_env, device="cpu") 
else:
    model.learn(total_timesteps=total_steps, progress_bar=True)
    model.save("ppo_node_env")
env = NodeEnv("../solcast2025.csv", step_s=STEP_S, stop_on_full_battery=False, discrete_action=args.discrete_action, discrete_state=args.discrete_state)
obs = env.reset()[0]
last_action = 0
time_m = 0
min_battery_level = 1
rewards = []
while True:
    action, _ = model.predict(obs)
    new_obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    #print("Obs:", obs, "Action:", action, "Reward:", reward)
    obs = new_obs
    battery_level = info["battery"]
    if battery_level < min_battery_level:
        min_battery_level = battery_level
    time_m += 1
    writer.add_scalars(f'check_info', {
    'panel_production': info["pv_solar"],
    'energy_consumption': info["current"],
    'battery_level': info["battery"],
    'processing_rate': action,
    #'angular_coefficient': obs[4]
}, time_m)
    if time_m == (60 * 24 * 14)// (STEP_S/60):  # 14 days
        print("Reached 14 days of simulation")
        env.device.show_plot(show=False, save=True, filename=f"14_days.png", additional_data=rewards)
    if done:  # 14 days
        print("Done")
        break
writer.close()

print(f"Lifetime: {env.time_s} s")
env.device.show_plot(show=False, save=True, filename="full")

print(f"Processed images: {env.get_amount_processed_images()}")
print(f"Efficiency: {env.device.get_energy_efficiency()}")
print(f"Total produced energy: {env.device.total_produced_energy_mah} mAh")
print(f"Total consumed energy: {env.device.total_consumed_energy_mah} mAh")
print(f"Total wasted energy: {env.device.total_wasted_energy_mah} mAh")
print(f"Uptime: {env.get_human_uptime()}")
print(f"Minimum battery level: {min_battery_level * 100:.2f}%")
print(f"Battery level: {env.device.get_battery_percentage()}")
