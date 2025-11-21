from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.utils import set_random_seed
from lib.env_bee_day import EnvBeeDay
from lib.utils import StateContent
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
from utils import *
from lib.solar.solar import Solar
from statistics import mean
import signal
import sys
from tqdm import tqdm
from max_images_pulp2 import find_optimal_list
from stable_baselines3.common.vec_env import SubprocVecEnv
from time import time
from thop import profile, clever_format
'''
Stato solo batteria, epochs 5, 3110 immagini, Update model on done or terminated

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",default=10000, required=False, type=int)
    parser.add_argument("--update_steps",default=512, required=False, type=int)
    parser.add_argument("--epochs",default=5, required=False, type=int)
    parser.add_argument("--alg", default="ppo", choices=["ppo","a2c","dqn","rec_ppo"])
    parser.add_argument("--use_solar", default=False, action="store_true")
    parser.add_argument("--use_month", default=False, action="store_true")
    parser.add_argument("--use_hour", default=False, action="store_true")
    parser.add_argument("--use_hour_minute", default=False, action="store_true")
    parser.add_argument("--use_minute", default=False, action="store_true")
    parser.add_argument("--use_day", default=False, action="store_true")
    parser.add_argument("--use_day_avg", default=False, action="store_true")
    parser.add_argument("--use_next_day", default=False, action="store_true")
    parser.add_argument("--use_real_forecast", default=False, action="store_true")
    parser.add_argument("--use_estimate_forecast", default=False, action="store_true")
    parser.add_argument("--use_estimate_single_forecast", default=False, action="store_true")
    parser.add_argument("--use_humidity", default=False, action="store_true")
    parser.add_argument("--use_cloud", default=False, action="store_true")
    parser.add_argument("--use_pressure", default=False, action="store_true")
    parser.add_argument("--use_sunset_time", default=False, action="store_true")
    parser.add_argument("--use_embed_day", default=False, action="store_true")
    parser.add_argument("--use_embed_next_day", default=False, action="store_true")
    parser.add_argument("--use_embed_prev_day", default=False, action="store_true")
    parser.add_argument("--use_quantize_day", default=False, action="store_true")
    parser.add_argument("--use_quantize_prev_day", default=False, action="store_true")
    parser.add_argument("--layer_width", default=64, type=int)
    parser.add_argument("--latent_size", default=24, type=int)
    parser.add_argument("--run_folder", default="../runs", type=str)
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--val",default=False, action="store_true")
    parser.add_argument("--model",default=None, type=str)
    parser.add_argument("--lr",type=float, default=0.0003)
    parser.add_argument("--lr_decay",default=None, choices=["exp","cos","lin","lin_half","lin_thirth"])
    parser.add_argument("--n_env", default=1, type=int, required=False)
    parser.add_argument("--term_days", default=1, type=int, required=False)
    parser.add_argument("--forecast_minutes",type=int, default=60)
    parser.add_argument("--prediction_accuracy", default=1.0, type=float)
    parser.add_argument("--choose_forecast",default=False, action="store_true")
    parser.add_argument("--test_year",type=int, default=2025)
    parser.add_argument("--test_freq",type=int, default=0)
    parser.add_argument("--train_days", type=int, default=30)
    parser.add_argument("--autostart", default=False, action="store_true")
    parser.add_argument("--start_thr",type=float, default=0.1)
    parser.add_argument("--prevision_noise", type=float, default=0.2)
    parser.add_argument("--use_images", default=False, action="store_true")
    args = parser.parse_args()
    SEED = 42

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    set_random_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if not args.val and args.run_folder == "../runs":
        if not os.path.exists(args.run_folder):
            os.mkdir(args.run_folder)
        folders = os.listdir(args.run_folder)
        folders = list(filter(lambda x:x.isdigit(), folders))
        if len(folders)==0:
            folders=[0]
        last_id = max(map(lambda x:int(x),folders))
        new_id = last_id+1
        args.run_folder = os.path.join(args.run_folder,str(new_id))
        
        def signal_handler(sig, frame):
            shutil.rmtree(args.run_folder)
            print(f"Run folder {args.run_folder} cleaned")
            sys.exit(0)
        signal.signal(signal.SIGTERM, signal_handler) #SIGTEM sent from tsp -k

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
    if args.use_hour_minute:
        state_content ^= StateContent.HOUR_MINUTE
    if args.use_sunset_time:
        state_content ^= StateContent.SUNSET_TIME
    if args.use_day_avg:
        state_content ^= StateContent.DAY_AVG
    if args.use_embed_day:
        state_content ^= StateContent.EMBEDDED_CURRENT_DAY
    if args.use_quantize_day:
        state_content ^= StateContent.QUANTIZED_DAY
    if args.use_quantize_prev_day:
        state_content ^= StateContent.QUANTIZED_PREV_DAY
    if args.use_embed_next_day:
        state_content ^= StateContent.EMBEDDED_NEXT_DAY
    if args.use_embed_prev_day:
        state_content ^= StateContent.EMBEDDED_PREV_NEXT_DAY
    if args.use_images:
        state_content ^= StateContent.IMAGES

    state_content ^= StateContent.BUFFER

    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    panel_area_m2 = 0.55*0.51 #m2
    efficiency = 0.1426
    max_power_w = 40 #W
    solar2024 = Solar("../solcast2024.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=args.prediction_accuracy)
    solar2025 = Solar(f"../solcast{args.test_year}.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=args.prediction_accuracy)

    start_hour = -1 if args.autostart else 7
    end_hour = -1 if args.autostart else 18
    vec_env = make_vec_env(lambda: EnvBeeDay(solar2024, 
                                            step_s=5*60,
                                            selected_day=1,
                                            start_hour=start_hour,
                                            end_hour=end_hour,
                                            acquistion_speed_fps=3,
                                            processing_speed_fps=4,
                                            seed=SEED, 
                                            state_content=state_content, 
                                            random_reset=True, 
                                            terminated_days=args.term_days, 
                                            forecast_time_m=args.forecast_minutes, 
                                            choose_forecast=args.choose_forecast,
                                            latent_size=args.latent_size,
                                            train_days=args.train_days,
                                            start_threshold=args.start_thr,
                                            prevision_noise_amount=args.prevision_noise),
                                        n_envs=args.n_env, vec_env_cls=SubprocVecEnv, seed=SEED)

    test_env = EnvBeeDay(  solar2025,
                        step_s=5*60,
                        selected_day=1,
                        start_hour=start_hour,
                        end_hour=end_hour,
                        acquistion_speed_fps=3,
                        processing_speed_fps=4,
                        seed=SEED,
                        state_content=state_content,
                        random_reset=False,
                        terminated_days=args.term_days,
                        forecast_time_m=args.forecast_minutes,
                        choose_forecast=args.choose_forecast,
                        latent_size=args.latent_size,
                        train_days=args.train_days,
                        start_threshold=args.start_thr,
                        prevision_noise_amount=args.prevision_noise)

    if args.lr_decay is None:
        lr = args.lr
        print(f"Fixed LR: {lr}")
    elif args.lr_decay == "exp":
        lr = exp_schedule(args.lr)
        print("Exponential LR descrese")
    elif args.lr_decay == "cos":   
        lr = cosine_schedule(args.lr)
        print("Cosine LR descrese")
    elif args.lr_decay == "lin":   
        lr = linear_schedule(args.lr)
        print("Linear LR descrese")
    elif args.lr_decay == "lin_half":   
        lr = linear_schedule(args.lr, start_from=0.5)
        print("Linear LR from half descrese")
    elif args.lr_decay == "lin_thirth":   
        lr = linear_schedule(args.lr, start_from=0.3)
        print("Linear LR from thirth descrese")
        

    alg_entry = None
    if args.alg == "ppo":
        alg_entry = partial(PPO,
                            n_epochs=args.epochs,
                            n_steps = args.update_steps,
                            batch_size=256,
                            vf_coef=0.5,
                            clip_range_vf=linear_schedule(0.5,0.3,end_value=0.1),
    )
    elif args.alg == "a2c":
        alg_entry = partial(A2C,
                            n_steps = args.update_steps,
                            ent_coef = 0.01
    )
    elif args.alg == "dqn":
        alg_entry = partial(DQN,
                            train_freq=args.update_steps)
    elif args.alg == "rec_ppo":
        alg_entry = partial(RecurrentPPO,
                            n_epochs=args.epochs,
                            n_steps = args.update_steps,
                            batch_size=256,
                            vf_coef=0.5,
                            clip_range_vf=linear_schedule(0.5,0.3,end_value=0.1),
                            )
        
    policy = "MlpLstmPolicy" if args.alg == "rec_ppo" else "MlpPolicy"
    extractor_kwargs = dict(
        normal_dim=test_env.observation_space.shape[0]-128, 
        forecast_dim=128, 
        latent_dim=args.latent_size
    )
    policy_kwargs = dict(
        net_arch=[args.layer_width,args.layer_width],
        features_extractor_class=ForecastEmbedded,
        features_extractor_kwargs=extractor_kwargs
    )
    print("Network", policy_kwargs)
    model = alg_entry(
        policy=policy,
        env = vec_env,
        device=device,
        learning_rate=lr,
        verbose=2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.run_folder,
        seed=SEED
    )
    print("Model created")

    if not args.val:
        try:
            callbacks = []
            eval_callback = EvalCallback(test_env)
            model.learn(total_timesteps=args.steps, progress_bar=False)#, callback=eval_callback)
            if args.run_folder is not None:
                model_path = os.path.join(args.run_folder,"last.pth")
                print("Model saved in",model_path)
                model.save(model_path)

            print("Trainign completed")
        except Exception as e:
            print(e)
            if args.run_folder:
                print("Cleaning run folder",args.run_folder)
                shutil.rmtree(args.run_folder)

    if args.run_folder:
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
        
    max_days = 120
    progress_bar = tqdm(range(max_days), leave=False)
    failed_days = 0
    needed_views = ["Battery","Solar","Hour", "Processing","Memory","Sunset","Hour Minute" ,"Opt Action","Opt Buffer", "Opt Batt", "Images"]
    optimal_processed_images_per_day: list[tuple[int,int]] = [] #[Opt, RL]
    fleassible_processed_images_per_day: list[tuple[int,int]] = [] #[Opt, RL]    
    processed_steps = 0
    is_fleassibles = []
    is_optimals = []
    inference_total_time_s = 0
    step_total_time_s = 0

    for d in progress_bar:
        test_obs, info = test_env.reset(options={"norandom":True, "day":d})
        fields = {}
        for f in info["fields"]:
            fields[f]=[]
        fields["Processing"]=[]
        datetimes = []
        start_battery = test_obs[0]
        processed_steps_per_day = 0
        captured_images = []

        while True:
            start_time = time()
            action,_ = model.predict(test_obs, deterministic=True)
            inference_total_time_s += (time()-start_time)
            start_time = time()
            test_obs, reward, done, terminated, info = test_env.step(action)
            step_total_time_s += (time() - start_time)
            processed_steps += 1
            processed_steps_per_day += 1
            if done and terminated==False:
                failed_days += 1
            for value,name in list(zip(test_obs, fields.keys())):
                fields[name].append(value.item())
            if "Processing" in fields.keys(): fields["Processing"].append(info["cons"])
            datetimes.append(info["time"])
            captured_images.append(info["images"])
            if done:
                break
        
        if done and terminated==False:
            # I need to get solars untill the end of the day
            time_s = test_env.time_s
            while True:
                solar_j = test_env.solar.get_solar_w(time_s)*test_env.step_size_s
                if solar_j > 0:
                    captured_images.append(test_env.acquisition_speed_fps*test_env.step_size_s)
                    fields["Solar"].append(solar_j/(test_env.solar.max_power_w*test_env.step_size_s))
                else:
                    break
                time_s += test_env.step_size_s
        
        solar_profiles = list(map(lambda x:x*test_env.solar.max_power_w*test_env.step_size_s, fields["Solar"]))
        #print("Battery starts a at",start_battery)
        optimal_prs, optimal_buff, optimal_batt, is_fleassible, is_optimal = find_optimal_list(
                solar_profiles,
                captured_images=captured_images,
                battery_start_percentage=start_battery,
                max_buffer=test_env.max_buffer_size,
                max_battery_j=test_env.battery_max_j,
                e_idle_for_step_j=test_env.energy_for_idle_step_j,
                e_processing_for_img_j=test_env.energy_for_image_processing_j
            )
        processed_imgs_opt = max(0,sum(optimal_prs))
        fields["Opt Action"] = list(map(lambda x:x/(test_env.processing_speed_fps*test_env.step_size_s), optimal_prs))
        fields["Opt Buffer"] = list(map(lambda x:x/test_env.max_buffer_size, optimal_buff))
        fields["Images"] = list(map(lambda x:x/(test_env.acquisition_speed_fps*test_env.step_size_s), captured_images))
        fields["Opt Batt"] = list(map(lambda x:x/(test_env.battery_max_j), optimal_batt))
        is_fleassibles.append(is_fleassible)
        is_optimals.append(is_optimal)

        if is_optimal:
            optimal_processed_images_per_day.append((processed_imgs_opt, test_env.processed_images))
        if is_fleassible:
            fleassible_processed_images_per_day.append((processed_imgs_opt, test_env.processed_images))

        if test_env.processed_images > processed_imgs_opt and is_optimal:
            print("RL is better than optimal solution | day",d,"RL",test_env.processed_images, "Opt",processed_imgs_opt)
            #break

        datetimes = list(map(lambda x:x.strftime("%H:%M"), datetimes))
        if args.run_folder:
            filename = os.path.join(args.run_folder,"test_images")
            os.makedirs(filename, exist_ok=True)
            filename = os.path.join(filename, f"{d}.png")
            save_plot(fields, filename, views=needed_views, x_values=None)

        progress_bar.set_postfix({
            "Opt":is_fleassible,
            "Processed RL":test_env.processed_images,
            "Processed Opt":processed_imgs_opt,
            "Failed":failed_days,
            "Steps RL": processed_steps_per_day,
            "Steps Opt": len(solar_profiles)
        })

    print("Only Fleassible")
    print("\tRL sum",sum(list(map(lambda x:x[1], fleassible_processed_images_per_day))))
    print("\tRL AVG",mean(list(map(lambda x:x[1], fleassible_processed_images_per_day))))
    print("\tILP sum",sum(list(map(lambda x:x[0], fleassible_processed_images_per_day))))
    print("\tILP AVG",mean(list(map(lambda x:x[0], fleassible_processed_images_per_day))))
    print("\tILP vs RF avg efficiency", mean(list(map(lambda x:x[1]/x[0], fleassible_processed_images_per_day))))
    print()
    print("Only Optimal")
    print("\tRL sum", sum(list(map(lambda x:x[1], optimal_processed_images_per_day))))
    print("\tRL avg", mean(list(map(lambda x:x[1], optimal_processed_images_per_day))))
    print("\tILP sum", sum(list(map(lambda x:x[0], optimal_processed_images_per_day))))
    print("\tILP avg", mean(list(map(lambda x:x[0], optimal_processed_images_per_day))))
    print("\tILP vs RF avg efficiency", mean(list(map(lambda x:x[1]/x[0], optimal_processed_images_per_day))))
    print()
    print("Early shutdown", failed_days)
    print("Processed steps",processed_steps)
    print("AVG inference time (ms)",1000*inference_total_time_s/processed_steps)
    print("AVG step time (ms)",1000*step_total_time_s/processed_steps)

    if args.run_folder:
        rl_results = list(map(lambda x:x[1], fleassible_processed_images_per_day))
        opt_results = list(map(lambda x:x[0], fleassible_processed_images_per_day))
        max_value_per_day = max(opt_results) 
        save_plot({
            "RL":list(map(lambda x:x/max_value_per_day,rl_results)),
            "Opt":list(map(lambda x:(x/max_value_per_day)+0.003,opt_results)),
            "Optimal":list(map(lambda x:x+0.006,is_optimals)),
            "Fleassible":list(map(lambda x:x+0.009,is_fleassibles))
        }, os.path.join(args.run_folder,"run_per_day.png"))

    
    print("Measuring flops")
    policy_module = model.policy.cpu()
    obs_shape = test_env.observation_space.shape
    dummy_input = torch.zeros(1, *obs_shape, dtype=torch.float32)
    inputs = (dummy_input,)
    policy_flops, policy_params = profile(policy_module, inputs=inputs, verbose=False)
    policy_flops_str, policy_params_str = clever_format([policy_flops, policy_params], "%.3f")
    print(f"FLOPs (forward pass): {policy_flops_str}")
    #print(f"Parameters: {policy_params_str}")

    if test_env.linear_mlp is not None:
        policy_module = test_env.linear_mlp.cpu()
        print(policy_module)
        dummy_input = torch.zeros(1, 128, dtype=torch.float32)
        inputs = (dummy_input,)
        embed_flops, embed_params = profile(policy_module, inputs=inputs, verbose=False)
        str_embed_flops, embed_params = clever_format([embed_flops, embed_params], "%.3f")
        print(f"Embedding FLOPs",str_embed_flops)
        total_flops = clever_format([policy_flops+embed_flops], "%.3f")
        print(f"Total FLOPs",total_flops)

    exit(0)

if __name__ == "__main__":
    main()
"""
processed_images = []
test_obs, info = test_env.reset(options={"norandom":True})
battery_levels, recharges, consumptions, times, forecast = [], [], [], [], []
battery_levels.append(test_obs[0])
done = False
rewards = []
is_plotted = False
plot_time = 60*60*24*31
unprocessed_images = 0
datetimes = []
fields = {}
for f in info["fields"]:
    fields[f]=[]
fields["Energy"]=[]

if test_env.choose_forecast:
    fields["start_time"]=[]
    fields["window_time_s"]=[]

# Remove fields
needed_views = ["Battery","Solar","Hour", "Energy","Hour Minute","Buffer","Sunset"]
print("Start time",test_env.time_s)
while True:
    # if processing_buffer:
    #     test_obs, reward, done, terminated, info = test_env.clear_buffer()
    #     if terminated or done:
    #         processing_buffer = False
    # else:
    action,_ = model.predict(test_obs, deterministic=True)
    test_obs, reward, done, terminated, info = test_env.step(action)
    #processing_buffer = terminated and test_env.buffer_length > 0
    rewards.append(reward)
    datetimes.append(info["time"])

    for value,name in list(zip(test_obs, fields.keys())):
        fields[name].append(value)
    if "Energy" in fields.keys(): fields["Energy"].append(info["cons"])
    if test_env.choose_forecast:
        fields["start_time"].append(info["forecast"][0])
        fields["window_time_s"].append(info["forecast"][1])

    #if steps % 1000 == 0: print("Processing day",test_env.get_uptime_s()//(60*60*24),"of 121 | Processed images",test_env.processed_images, end="\r")
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
        datetimes = list(map(lambda x:x.strftime("%H:%M"), datetimes))
        save_plot(fields, img_file, needed_views, x_values=datetimes)
        for f in fields.keys():
            fields[f].clear()
    if done or test_env.get_uptime_s() >= 120*24*60*60:
        print()
        print("BATTERY DEPLETED!",done,terminated)
        break
    if terminated:
        image_folder = os.path.join(args.run_folder,"images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image.jpg")
        datetimes = list(map(lambda x:x.strftime("%H:%M"), datetimes))
        save_plot(fields, img_file, needed_views, x_values=datetimes)
        for f in fields.keys():
            fields[f].clear()
        break

print("End time",test_env.time_s)
print("Time reached", test_env.get_human_uptime(), test_env.get_uptime_s())
#print("Final reward", sum(rewards),len(rewards))
print("Processed images", test_env.processed_images)
print("Buffer images",test_env.buffer_length)
processed_images.append(test_env.processed_images)

print(processed_images)
"""