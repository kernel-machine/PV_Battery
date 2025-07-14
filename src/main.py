from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from torch.utils.tensorboard import SummaryWriter
from lib.env import NodeEnv
from lib.pid_controller import PIDController
import random
import datetime
import argparse

args = argparse.ArgumentParser()
args.add_argument("--short_training", default=False, action="store_true", help="Set to have a shorter training")
args.add_argument("--n_env", type=int, default=4, help="Number of environment to train in parallel")
args.add_argument("--val", default=False, action="store_true", help="Validate the model without training")
args.add_argument("--discrete_action", default=False, action="store_true", help="Use discrete actions instead of continuous")
args.add_argument("--discrete_state", default=False, action="store_true", help="Use discrete state space instead of continuous")
args.add_argument("--alg",type=str,required=True,help="RL algoirthm name", choices=["dqn","ppo","a2c"])
args.add_argument("--batch_size",type=int, default=128, help="Batch size for model update")
args.add_argument("--exploration_fraction", type=float, default=0.1, help="Exploration fraction")
args.add_argument("--truncate_after", type=int, default=30, help="Truncate episode after X days")
args.add_argument("--incentive_factor", type=float, default=0.3, help="Incentive factor for reward") #depends by the reward
args.add_argument("--img_name",type=str, default="")
args.add_argument("--pid",default=False, action="store_true", help="Use PID controller instead of RL")
args.add_argument("--ppo_epochs",default=10, type=int, help="Epochs for PPO")
args = args.parse_args()
SEED = 1234
STEP_S = 60*5 # Step size in seconds
random.seed(SEED)
run_identifier = f"{args.alg}_b{args.batch_size}_n{args.n_env}_{args.img_name}"

# log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir=log_dir)

if not args.pid and not args.val:
    if args.discrete_state:
        env = NodeEnv("../solcast2024.csv", step_s=STEP_S, stop_on_full_battery=True, discrete_action=args.discrete_action, discrete_state=args.discrete_state, truncate_alter_d=args.truncate_after, incentive_factor=args.incentive_factor)
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
            env_kwargs={"csv_solar_data_path": "../solcast2024.csv", "step_s": STEP_S, "stop_on_full_battery": True, "discrete_state": args.discrete_state,"discrete_action": args.discrete_action, "truncate_alter_d" : args.truncate_after, "incentive_factor":args.incentive_factor},
        )
        if args.alg == "ppo":
            model = PPO(
                "MlpPolicy",
                vec_env,
                #tensorboard_log=log_dir,
                verbose=2,
                device="cpu",
                #n_steps=(24*60*60)//STEP_S, # 1 Day
                seed=SEED,
                batch_size=args.batch_size,
                n_epochs=args.ppo_epochs
            # policy_kwargs=dict(log_std_init=-2),
            )
        elif args.alg == "a2c":
            model = A2C(
                "MlpPolicy",
                vec_env,
                n_steps=(24*60*60)//STEP_S, # 1 Day
                verbose=2,
                seed=SEED,
                device="cpu",
            )
        elif args.alg == "dqn":
            model = DQN(
                "MlpPolicy",
                vec_env,
                verbose=2,
                learning_starts=10000,
                batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction,
                #exploration_initial_eps=0.8,
                #exploration_final_eps=0.01,
                seed=SEED
                )
        days = 365
        total_steps = (days * 24 * 60 * 60 // STEP_S) * args.n_env  # Total steps for training

        # if args.short_training:
        #     callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)
        #     eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
        # else:
        #     eval_callback = None

    if not args.val:
        model.learn(total_timesteps=total_steps, progress_bar=True, callback=None)
        model.save(f"runs/ppo_node_env_{run_identifier}")

if True: #val
    env = NodeEnv("../solcast2025.csv", step_s=STEP_S, stop_on_full_battery=False, discrete_action=args.discrete_action, discrete_state=args.discrete_state, incentive_factor=args.incentive_factor, truncate_alter_d=args.truncate_after)
    
    if args.pid:
        controller = PIDController(kp=100, ki=0, dt=0)
        battery_setpoint = 0.2  # Desired battery level (80%)
    else:
        model = PPO.load(f"runs/ppo_node_env_{run_identifier}", verbose=1) 
    # model.set_parameters(f"runs/ppo_node_env_{run_identifier}")

    obs = env.reset(seed=SEED, options={"norandom":True})[0]

    min_battery_level = 1
    rewards = []
    days_14_reached = False

    while True:
        if args.pid:
            error = battery_setpoint - env.device.get_battery_percentage()
            control_output = -controller.update(error)
            action = 1 if control_output > 0 else 0
            action = [action]  # Ensure action is a list for the environment
            # Ensure action is a list for the environment
        else:
            action, _ = model.predict(obs)
        
        new_obs, reward, done, _, __ = env.step(action)
        rewards.append(reward)
        if env.get_uptime_s() % STEP_S == 0:
            print(env.get_uptime_s()//60," | Obs:", obs, "Action:", action, "Reward:", reward, end="\r")
        obs = new_obs

        battery_level = env.device.get_battery_percentage()    
        if battery_level < min_battery_level:
            min_battery_level = battery_level

        if not days_14_reached and env.get_uptime_s() > (60 * 24 * 60 * 3):  # 14 days
            print("Reached 14 days of simulation")
            env.device.show_plot(show=False, save=True, filename=f"img/14_days_model_{run_identifier}.png", additional_data=rewards)
            days_14_reached = True
        if done:
            env.device.show_plot(show=False, save=True, filename=f"img/14_days_model_{run_identifier}_complete.png", additional_data=rewards, start_from=60 * 24 * 60 * 118)
            break


    print(f"Lifetime: {env.time_s} s")
    env.device.show_plot(show=False, save=True, filename="full")

    print(f"Processed images: {env.get_amount_processed_images()}")
    print(f"Efficiency: {env.device.get_energy_efficiency()}")
    print(f"Total produced energy: {env.device.total_produced_energy_wh} mAh")
    print(f"Total consumed energy: {env.device.total_consumed_energy_wh} mAh")
    print(f"Total wasted energy: {env.device.total_wasted_energy_wh} mAh")
    print(f"Uptime: {env.get_human_uptime()}")
    print(f"Minimum battery level: {min_battery_level * 100:.2f}%")
    print(f"Battery level: {env.device.get_battery_percentage()}")
    print(f"Number of steps: {env.number_of_steps} {env.number_of_high} {env.number_of_low}")
