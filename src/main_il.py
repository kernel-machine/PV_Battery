from ilp_solver import find_optimal_list
from utils import save_plot
from lib.solar.solar import Solar
from random import randint, seed
from math import ceil
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import torch
import argparse
import os
from tqdm import tqdm

PANEL_AREA_M2 = 0.55 * 0.51  # m2
EFFICIENCY = 0.1426
MAX_POWER_W = 40  # W
DELTA_T = 5 * 60
START_HOUR = 8
END_HOUR = 17
PROCESSING_SPEED = 4
MAX_BATTERY_J = 6.6*3.7*3600
ENERGY_IDLE_STEP_J = 75
ENERGY_FOR_IMAGE_J = 3
INPUT_SIZE = 109+1
OUTPUT_SIZE = 109

class Model(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, layer_width:int = 512):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, layer_width*2),
            torch.nn.BatchNorm1d(layer_width * 2),
            torch.nn.ReLU(),

            torch.nn.Linear(layer_width * 2, layer_width * 2),
            torch.nn.BatchNorm1d(layer_width * 2),
            torch.nn.ReLU(),

            torch.nn.Linear(layer_width * 2, layer_width),
            torch.nn.BatchNorm1d(layer_width),
            torch.nn.ReLU(),

            torch.nn.Linear(layer_width,output_size),
            #torch.nn.Sigmoid()
        )


    def forward(self, x):
        y = self.layers(x)
        y = torch.clamp(y, 0, 1)
        return y

def simulate(start_battery_j:float,
             max_battery_j:float,
             idle_consumption_j:float,
             full_power_energy_j:float,
             solar_energy_j:list[float],
             actions:list[float]
             ) -> list[float]:
    
    battery_steps = []
    battery_steps.append(start_battery_j)
    turned_off = False

    for index in range(len(solar_energy_j)):
        solar = solar_energy_j[index]
        action = actions[index]

        if turned_off:
            new_battery = 0
        else:
            new_battery = min(max_battery_j, battery_steps[-1] + solar - action * full_power_energy_j - idle_consumption_j)
            if new_battery < 0:
                turned_off = True
                new_battery = 0
        battery_steps.append(new_battery)
    
    return battery_steps

test_cache = {}
def test(model:torch.nn.Module, batch_size:int, plots:bool = False):
    model.eval()
    solar = Solar(
        "../solcast2025.csv",
        scale_factor=PANEL_AREA_M2 * EFFICIENCY,
        max_power=MAX_POWER_W,
        enable_cache=True,
    )
    start_battery_level = 0.25
    max_days = 120
    batch_inputs = []
    target_actions = []
    loss = torch.nn.SmoothL1Loss(beta=1)
    losses_values = []
    progress_bar = tqdm(range(max_days), desc="Validation")
    actions = []
    for d in progress_bar:
        start_time_s = (d*24+START_HOUR)*60*60
        end_time_s = (d*24+END_HOUR)*60*60+300
        start_slot = start_time_s//DELTA_T
        T_max = (end_time_s//DELTA_T)
        solar_profiles = list(range(start_slot, T_max))
        solar_profiles = list(map(lambda x:solar.get_solar_w(x*DELTA_T)*DELTA_T, solar_profiles))
        if sum(solar_profiles) > ENERGY_IDLE_STEP_J*len(solar_profiles):
            if str(d) not in test_cache.keys():
                optimal_actions, is_optimal = find_optimal_list(solar_profiles,
                                                    battery_start_percentage=start_battery_level,
                                                    max_battery_j=MAX_BATTERY_J,
                                                    e_idle_for_step_j=ENERGY_IDLE_STEP_J,
                                                    e_processing_for_img_j=ENERGY_FOR_IMAGE_J)
                test_cache[str(d)]= (optimal_actions, is_optimal)
            else:
                optimal_actions, is_optimal = test_cache[str(d)]
            
            if is_optimal: #If problem is solved
                solar_profiles = list(map(lambda x:x/(solar.max_power_w*DELTA_T), solar_profiles))
                optimal_actions = list(map(lambda x:x/(PROCESSING_SPEED*DELTA_T), optimal_actions))
                solar_profiles.append(start_battery_level)

                batch_inputs.append(solar_profiles)
                target_actions.append(optimal_actions)

        if len(target_actions) == batch_size:
                batch_inputs = torch.tensor(batch_inputs)
                target_actions = torch.tensor(target_actions)

                with torch.no_grad():
                    out = model(batch_inputs)
                    l = loss(out, target_actions)
                    losses_values.append(l.item())
                    progress_bar.set_postfix({
                        "loss":l.item()
                    })
                    if plots:
                        for b in range(batch_size):
                            actions.append({
                                "profile":batch_inputs[b][:-1].numpy().tolist(),
                                "start_battery":batch_inputs[b][-1].item(),
                                "target":target_actions[b].numpy().tolist(),
                                "action":out[b].numpy().tolist()
                            })
                target_actions = []
                batch_inputs = []

    if plots:
        for index, e in enumerate(actions):
            batt_optimal = simulate(
                start_battery_j=e["start_battery"]*MAX_BATTERY_J,
                max_battery_j=MAX_BATTERY_J,
                idle_consumption_j=ENERGY_IDLE_STEP_J,
                full_power_energy_j=ENERGY_FOR_IMAGE_J*DELTA_T,
                solar_energy_j=list(map(lambda x:x*solar.max_power_w*DELTA_T,e["profile"])),
                actions=list(map(lambda x:x*PROCESSING_SPEED,e["target"]))
            )
            batt_ai = simulate(
                start_battery_j=e["start_battery"]*MAX_BATTERY_J,
                max_battery_j=MAX_BATTERY_J,
                idle_consumption_j=ENERGY_IDLE_STEP_J,
                full_power_energy_j=ENERGY_FOR_IMAGE_J*DELTA_T,
                solar_energy_j=list(map(lambda x:x*solar.max_power_w*DELTA_T,e["profile"])),
                actions=list(map(lambda x:x*PROCESSING_SPEED,e["action"]))

            )
            fields = {
                "Batt_opt":list(map(lambda x:x/MAX_BATTERY_J, batt_optimal)),
                "Batt_AI":list(map(lambda x:x/MAX_BATTERY_J, batt_ai)),
                "Solar":e["profile"],
                "Optimal":e["target"],
                "AI":e["action"],
            }
            custom_alpha = {
                "Optimal":0.3,
                "AI":0.3
            }
            save_plot(fields, path=f"../val_il/day_{index}.jpg", custom_alpha=custom_alpha)


    return mean(losses_values)


def train(epochs:int, batch_size:int, learing_rate:float, device:torch.device):
    writer = SummaryWriter(log_dir=f"runs/il_bs{args.batch_size}_lr{args.lr}_v2")
    solar = Solar(
        "../solcast2024.csv",
        scale_factor=PANEL_AREA_M2 * EFFICIENCY,
        max_power=MAX_POWER_W,
        enable_cache=True,
    )

    model = Model(INPUT_SIZE, OUTPUT_SIZE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learing_rate)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss = torch.nn.SmoothL1Loss(beta=1)
    for epoch in range(epochs):
        batch_inputs = []
        target_actions = []
        losses_values = []
        model.train()
        progress_bar = tqdm(range(1,350), desc=f"Training epoch: {epoch+1}/{epochs}")
        for d in progress_bar:
            start_time_s = (d*24+START_HOUR)*60*60
            end_time_s = (d*24+END_HOUR)*60*60+300
            start_slot = start_time_s//DELTA_T
            T_max = (end_time_s//DELTA_T)
            solar_profiles = list(range(start_slot, T_max))
            solar_profiles = list(map(lambda x:solar.get_solar_w(x*DELTA_T)*DELTA_T, solar_profiles))
            
            random_shift = randint(50,150)/100
            solar_profiles = list(map(lambda x:max(0,x*random_shift), solar_profiles))

            time_augmentation = False
            if time_augmentation:
                len_solar_profiles = len(solar_profiles)
                split_index = randint(len_solar_profiles//4, 3*len_solar_profiles//4)
                multiplies_left = randint(60,100)/100
                multiplies_right = randint(80,130)/100
                new_profiles = [0 for _ in solar_profiles]
                for i in range(0, split_index):
                    value = solar_profiles[i]
                    new_index = int(i*multiplies_left)
                    new_profiles[new_index]=value
                for i in range(split_index, len_solar_profiles):
                    value = solar_profiles[i]
                    new_index = int(i*multiplies_right)
                    new_index = min(len_solar_profiles,max(split_index, new_index))
                    new_profiles[new_index]=value

                # Fill holes with linear interpolation
                for i in range(len_solar_profiles):
                    if new_profiles[i]==0:
                        prev_element = 0
                        next_element = 0
                        for k in range(i,0,-1):
                            if new_profiles[k]>0:
                                prev_element = new_profiles[k]
                                break
                        for k in range(i,len_solar_profiles):
                            if new_profiles[k]>0:
                                next_element = new_profiles[k]
                                break
                        new_profiles[i] = prev_element + (next_element-prev_element)/2
                solar_profiles = new_profiles
                

            if sum(solar_profiles) > ENERGY_IDLE_STEP_J*len(solar_profiles):
                # optimal exists

                battery_level = randint(8,50)/100
                optimal_actions, _ = find_optimal_list(solar_profiles,
                                                            battery_start_percentage=battery_level,
                                                            max_battery_j=MAX_BATTERY_J,
                                                            e_idle_for_step_j=ENERGY_IDLE_STEP_J,
                                                            e_processing_for_img_j=ENERGY_FOR_IMAGE_J)

                solar_profiles = list(map(lambda x:x/(solar.max_power_w*DELTA_T), solar_profiles))
                optimal_actions = list(map(lambda x:x/(PROCESSING_SPEED*DELTA_T), optimal_actions))
                solar_profiles.append(battery_level)

                batch_inputs.append(solar_profiles)
                target_actions.append(optimal_actions)

                #print(f"Buffer filled {len(batch_inputs)}/{BATCH_SIZE}")

            if len(target_actions) == batch_size:
                batch_inputs = torch.tensor(batch_inputs).to(device)
                target_actions = torch.tensor(target_actions).to(device)

                out = model(batch_inputs)
                l = loss(out, target_actions)
                losses_values.append(l.cpu().item())
                progress_bar.set_postfix({
                    "loss":l.cpu().item()
                })
                l.backward()
                optimizer.step()

                for i in range(batch_size):
                    solar_profiles = batch_inputs[i][:-1].cpu().numpy().tolist()
                    optimal = target_actions[i].cpu().numpy().tolist()
                    ai_out = out[i].cpu().clip(min=0, max=1).detach().numpy().tolist()
                    batt_optimal = simulate(
                        start_battery_j=float(batch_inputs[i][-1])*MAX_BATTERY_J,
                        max_battery_j=MAX_BATTERY_J,
                        idle_consumption_j=ENERGY_IDLE_STEP_J,
                        full_power_energy_j=ENERGY_FOR_IMAGE_J*DELTA_T,
                        solar_energy_j=list(map(lambda x:x*solar.max_power_w*DELTA_T,solar_profiles)),
                        actions=list(map(lambda x:x*PROCESSING_SPEED,optimal))
                    )
                    batt_ai = simulate(
                        start_battery_j=float(batch_inputs[i][-1])*MAX_BATTERY_J,
                        max_battery_j=MAX_BATTERY_J,
                        idle_consumption_j=ENERGY_IDLE_STEP_J,
                        full_power_energy_j=ENERGY_FOR_IMAGE_J*DELTA_T,
                        solar_energy_j=list(map(lambda x:x*solar.max_power_w*DELTA_T,solar_profiles)),
                        actions=list(map(lambda x:x*PROCESSING_SPEED,ai_out))

                    )
                    fields = {
                        "Batt_opt":list(map(lambda x:x/MAX_BATTERY_J, batt_optimal)),
                        "Batt_AI":list(map(lambda x:x/MAX_BATTERY_J, batt_ai)),
                        "Solar":solar_profiles,
                        "Optimal":optimal,
                        "AI":ai_out,
                    }
                    custom_alpha = {
                        "Optimal":0.3,
                        "AI":0.3
                    }
                    save_plot(fields, path=f"../imitation_learning/batch_{i}.jpg", custom_alpha=custom_alpha)

                OUT_SUMs = (PROCESSING_SPEED * DELTA_T * out.sum(dim=1)).detach().cpu().numpy().tolist()
                TARGET_SUMs = (PROCESSING_SPEED * DELTA_T * target_actions.sum(dim=1)).detach().cpu().numpy().tolist()
                OUT_SUMs = [int(round(x)) for x in OUT_SUMs]
                TARGET_SUMs = [int(round(x)) for x in TARGET_SUMs]

                batch_inputs = []
                target_actions = []

        writer.add_scalar("Loss/Train",mean(losses_values),epoch)
        val_loss = test(model, args.batch_size)
        writer.add_scalar("Loss/Val",val_loss,epoch)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        decay.step()

        model_path = os.path.join(writer.log_dir,"last.pth")
        torch.save(model.state_dict(), model_path)

    writer.close()
if __name__ == "__main__":
    seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer_width", type=int, default=512)
    parser.add_argument("--val",type=str, default=None)
    parser.add_argument("--gpu", default=False, action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu else "cpu")
    batch_size = args.batch_size
    lr = args.lr
    if args.val is not None:
        model = Model(INPUT_SIZE, OUTPUT_SIZE)
        model.load_state_dict(torch.load(args.val))
        model.to(device)
        test(model, batch_size, plots=True, device=device)
    else:
        train(args.epochs, batch_size, lr, device=device)