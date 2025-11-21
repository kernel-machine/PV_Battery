import pulp as pl
from lib.solar.solar import Solar
#from max_images import save_plot
from utils import save_plot
from random import randint
import os
import tempfile
from tqdm import tqdm
from statistics import mean

PROCESSING_FPS = 4
ACQUISITION_FPS = 3
MAX_BATTERY_J = 6.6 * 3.7 * 3600
DELTA_T = 5 * 60
E_IDLE_J = 5 * 0.05 * DELTA_T
E_PROCESSING_FOR_IMG_J = 10/PROCESSING_FPS
I = DELTA_T * ACQUISITION_FPS
MAX_PR = DELTA_T * PROCESSING_FPS
MAX_BUFFER = DELTA_T * ACQUISITION_FPS * 20

def find_optimal_list(  solar_profile:list[float], 
                        captured_images:list[int],
                        processing_speed_fps:int = 4,
                        acquisition_speed_fps:int = 3,
                        max_battery_j:float = 6.6 * 3.7 * 3600,
                        battery_start_percentage:float = 0.3,
                        max_buffer:int = 15000,
                        e_idle_for_step_j:int = 75,
                        e_processing_for_img_j:int = 3,
                        delta_t:int = 5*60,
                        print_stats:bool = False,
                      ) -> tuple[list[int],list[int],list[int], bool, bool]:

    I = delta_t * acquisition_speed_fps
    MAX_PR = delta_t * processing_speed_fps

    M_batt = 10 * max_battery_j  # big-M, deve essere molto più grande della batteria
    M_buff = 10 * max_buffer

    start_slot = 0
    T_max = len(solar_profile)

    # Problema
    prob = pl.LpProblem("myProblem", pl.LpMaximize)

    # Variabili decisionali
    # P[t] = number of images processed in slot t (integer or continuous)
    P = pl.LpVariable.dicts("P", range(start_slot, T_max), lowBound=0, upBound=MAX_PR, cat="Integer")

    B = pl.LpVariable.dicts("B", range(start_slot-1, T_max), lowBound=0, cat=pl.LpContinuous) #B(t) Battery
    U = pl.LpVariable.dicts("U", range(start_slot-1, T_max), lowBound=0, cat=pl.LpContinuous) #U(t) Buffer

    # Variabili binarie per Big-M
    y_batt = pl.LpVariable.dicts("y_batt", range(start_slot, T_max), cat="Binary")
    y_buff = pl.LpVariable.dicts("y_buff", range(start_slot, T_max), cat="Binary")

    # Condizione iniziale
    prob += B[start_slot-1] == max_battery_j * battery_start_percentage, "Initial battery"
    prob += U[start_slot-1] == 0

    # Vincoli ricorsivi con Big-M per linearizzare i min()
    for t in range(start_slot, T_max):
        # energy per image in J (E_PROCESSING_J already defined as J/image)
        theoretical_battery = B[t - 1] + solar_profile[t] - e_idle_for_step_j - e_processing_for_img_j * P[t]

        # B[t] = min(theoretical_battery, MAX_BATTERY_J) via Big-M
        prob += B[t] <= theoretical_battery, f"Batt_ub_theoretical_{t}"
        prob += B[t] <= max_battery_j, f"Batt_ub_max_{t}"
        prob += B[t] >= theoretical_battery - M_batt * (1 - y_batt[t]), f"Batt_lb_theoretical_{t}"
        prob += B[t] >= max_battery_j - M_batt * y_batt[t], f"Batt_lb_max_{t}"

        # Buffer: images available at start + arrivals - processed images
        theoretical_buffer = U[t-1] + captured_images[t] - P[t]

        # U[t] = min(theoretical_buffer, MAX_BUFFER) via Big-M
        prob += U[t] <= theoretical_buffer, f"Buff_ub_theoretical_{t}"
        prob += U[t] <= max_buffer, f"Buff_ub_max_{t}"
        prob += U[t] >= theoretical_buffer - M_buff * (1 - y_buff[t]), f"Buff_lb_theoretical_{t}"
        prob += U[t] >= max_buffer - M_buff * y_buff[t], f"Buff_lb_max_{t}"

        # Cannot process more images than available this slot
        prob += P[t] <= U[t-1] + captured_images[t], f"Processing_availability_{t}"

    # Funzione obiettivo (massimizza la somma degli x_t e b_t)
    prob += pl.lpSum(P[t] for t in range(start_slot, T_max)) - pl.lpSum(U[t] for t in range(start_slot, T_max)), "MaxProcessedImages"

    # Risolvi
    log_file = tempfile.NamedTemporaryFile()
    prob.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=2*60, logPath=log_file.name))
    # Leggo il contenuto del log
    with open(log_file.name, "r") as f:
        log_text = f.read()
    is_optimal = "Result - Optimal solution found" in log_text
    not_fleassible = "Problem is infeasible" in log_text
    os.remove(log_file.name)
    #print(f"optimal",is_optimal,"fleassibe",not not_fleassible)
    #print("Status:", pl.LpStatus[prob.status], is_optimal)

    if print_stats:
        # Risultati
        print("Status:", pl.LpStatus[prob.status])
        #print("Final battery", B[T_max - 1].value())
        # for t in range(1, T_max):
        #     print(f"x[{t}] = {x[t].value()},   E[{t}] = {E[t].value()},   y[{t}] = {y[t].value()},   z[{t}] = {z[t].value()}")
        battery_levels = [1 if y_batt[t].value()==0 else B[t].value()/max_battery_j for t in range(start_slot,T_max)]
        datetimes = None#[solar.get_datetime(t*delta_t).strftime("%H:%M") for t in range(start_slot,T_max)]
        fields = {
            "Battery":battery_levels,
            "Solar": [solar_profile[t]/(MAX_POWER_W*delta_t) for t in range(start_slot,T_max)],
            "Energy": [P[t].value()/MAX_PR for t in range(start_slot,T_max)],
            "Buffer": [U[x].value()/max_buffer for x in range(start_slot,T_max)]
        }
        for i in fields.keys():
            print(i, len(fields[i]))
        save_plot(fields,"max_images_pulp.png", x_values=datetimes)

    return [int(P[t].value()) for t in range(start_slot,T_max)], [U[t].value() for t in range(start_slot,T_max)], [B[t].value() for t in range(start_slot,T_max)], not not_fleassible, is_optimal

def find_optimal2(start_day_:int, solar:Solar, start_hour:int = -1, end_hour:int=-1, start_battery_perc:float=0.3):
    DELTA_T = 5 * 60
    start_time_s = (start_day_*24+(0 if start_hour < 1 else start_hour))*60*60
    end_time_s = (start_day_*24+(23 if end_hour < 1 else end_hour))*60*60+300
    start_slot = start_time_s//DELTA_T
    T_max = (end_time_s//DELTA_T)
    solar_profiles_j = list(range(start_slot, T_max))
    solar_profiles_j = list(map(lambda x:solar.get_solar_w(x*DELTA_T)*DELTA_T, solar_profiles_j))
    if start_hour < 0:
        battery_j = 0
        start_index = 0
        for i, v in enumerate(solar_profiles_j):
            battery_j += min(max(0,v),solar.max_power_w*DELTA_T)
            if battery_j/MAX_BATTERY_J >= start_battery_perc:
                start_index = i+1
                break
        solar_profiles_j = solar_profiles_j[start_index:]
    if end_hour < 0:
        end_index = 0
        for i, v in enumerate(solar_profiles_j):
            if v <= 0:
                end_index = i+2
                break
        solar_profiles_j = solar_profiles_j[:end_index]

    processed_steps = len(solar_profiles_j)
    return find_optimal_list(solar_profiles_j, print_stats=True, battery_start_percentage=start_battery_perc, e_idle_for_step_j=750, e_processing_for_img_j=7.5/4)

def find_optimal(start_day_:int, solar:Solar, start_hour:int = 6, end_hour:int=18):
    # Parametri
    start_time_s = (start_day_*24+start_hour)*60*60
    end_time_s = (start_day_*24+end_hour)*60*60+300
    start_slot = start_time_s//DELTA_T
    T_max = (end_time_s//DELTA_T)
    M_batt = 10 * MAX_BATTERY_J  # big-M, deve essere molto più grande della batteria
    M_buff = 10 * MAX_BUFFER

    def E_s(t):
        return solar.get_solar_w(t * DELTA_T) * DELTA_T

    # Problema
    prob = pl.LpProblem("myProblem", pl.LpMaximize)

    # Variabili decisionali
    # P[t] = number of images processed in slot t (integer or continuous)
    P = pl.LpVariable.dicts("P", range(start_slot, T_max), lowBound=0, upBound=MAX_PR, cat="Integer")

    B = pl.LpVariable.dicts("B", range(start_slot-1, T_max), lowBound=0, cat=pl.LpContinuous) #B(t) Battery
    U = pl.LpVariable.dicts("U", range(start_slot-1, T_max), lowBound=0, cat=pl.LpContinuous) #U(t) Buffer

    # Variabili binarie per Big-M
    y_batt = pl.LpVariable.dicts("y_batt", range(start_slot, T_max), cat="Binary")
    y_buff = pl.LpVariable.dicts("y_buff", range(start_slot, T_max), cat="Binary")

    # Condizione iniziale
    prob += B[start_slot-1] == MAX_BATTERY_J * 0.1, "Initial battery"
    prob += U[start_slot-1] == 0

    # Vincoli ricorsivi con Big-M per linearizzare i min()
    for t in range(start_slot, T_max):
        # energy per image in J (E_PROCESSING_J already defined as J/image)
        theoretical_battery = B[t - 1] + E_s(t) - E_IDLE_J - E_PROCESSING_FOR_IMG_J * P[t]

        # B[t] = min(theoretical_battery, MAX_BATTERY_J) via Big-M
        prob += B[t] <= theoretical_battery, f"Batt_ub_theoretical_{t}"
        prob += B[t] <= MAX_BATTERY_J, f"Batt_ub_max_{t}"
        prob += B[t] >= theoretical_battery - M_batt * (1 - y_batt[t]), f"Batt_lb_theoretical_{t}"
        prob += B[t] >= MAX_BATTERY_J - M_batt * y_batt[t], f"Batt_lb_max_{t}"

        # Buffer: images available at start + arrivals - processed images
        theoretical_buffer = U[t-1] + I - P[t]

        # U[t] = min(theoretical_buffer, MAX_BUFFER) via Big-M
        prob += U[t] <= theoretical_buffer, f"Buff_ub_theoretical_{t}"
        prob += U[t] <= MAX_BUFFER, f"Buff_ub_max_{t}"
        prob += U[t] >= theoretical_buffer - M_buff * (1 - y_buff[t]), f"Buff_lb_theoretical_{t}"
        prob += U[t] >= MAX_BUFFER - M_buff * y_buff[t], f"Buff_lb_max_{t}"

        # Cannot process more images than available this slot
        prob += P[t] <= U[t-1] + I, f"Processing_availability_{t}"

    # Funzione obiettivo (massimizza la somma degli x_t e b_t)
    prob += pl.lpSum(P[t] for t in range(start_slot, T_max)), "MaxProcessedImages"

    # Risolvi
    prob.solve()

    # Risultati
    print("Stato:", pl.LpStatus[prob.status])
    print("Final battery", B[T_max - 1].value())
    # for t in range(1, T_max):
    #     print(f"x[{t}] = {x[t].value()},   E[{t}] = {E[t].value()},   y[{t}] = {y[t].value()},   z[{t}] = {z[t].value()}")

    battery_levels = [1 if y_batt[t].value()==0 else B[t].value()/MAX_BATTERY_J for t in range(start_slot,T_max)]
    datetimes = [solar.get_datetime(t*DELTA_T).strftime("%H:%M") for t in range(start_slot,T_max)]
    fields = {
        "Battery":battery_levels,
        "Solar": [E_s(t)/(MAX_POWER_W*DELTA_T) for t in range(start_slot,T_max)],
        "Energy": [P[t].value()/MAX_PR for t in range(start_slot,T_max)],
        "Buffer": [U[x].value()/MAX_BUFFER for x in range(start_slot,T_max)]
    }
    for i in fields.keys():
        print(i, len(fields[i]))
    save_plot(fields,"max_images_pulp.png", x_values=datetimes)

    #print("Images on the buffer",sum([b[t].value() for t in range(start_slot,T_max)]))
    #print("Processable images", E[T_max - 1].value()//E_PROCESSING_J)
    #print("Instant processed images",sum([x[t].value() for t in range(start_slot,T_max)]))
    #print("Unprocessed instant images",len(list(filter(lambda x:x==0,[x[t].value() for t in range(start_slot,T_max)]))))
    #print("Processed later from the buffer",sum([b[t].value() for t in range(start_slot,T_max)]))
    elapted_time_s = end_time_s-start_time_s
    h,m=elapted_time_s//(60*60),elapted_time_s%(60*60)//60
    print("Uptime H:",h,"M:",m, "Start time (s)",start_time_s,"End time (s)", end_time_s, "Up for",end_time_s-start_time_s,"s")
    #print("Number of images",len(x))
    return sum([P[t].value() for t in range(start_slot,T_max)]),[P[t].value() for t in range(start_slot,T_max)]


if __name__ == "__main__":
    PANEL_AREA_M2 = 0.55 * 0.51  # m2
    EFFICIENCY = 0.1426
    MAX_POWER_W = 40  # W
    solar = Solar(
        "../solcast2025.csv",
        scale_factor=PANEL_AREA_M2 * EFFICIENCY,
        max_power=MAX_POWER_W,
        enable_cache=True,
    )
    TEST_ALL_DAYS = False
    max_days = 120
    processed_steps = 0
    processed_images = []
    if TEST_ALL_DAYS:
        progress_bar = tqdm(range(0,max_days,15))
        for d in progress_bar:
            processing_rates, _ = find_optimal2(d, solar, start_hour=-1, end_hour=-1, start_battery_perc=0.05)
            processed_steps += len(processing_rates)
            processed_images.append(sum(processing_rates))
            print(f"Day {d}, steps: {len(processing_rates)}, processing rates: {sum(processing_rates)}")
    else:
        processing_rates, _ = find_optimal2(2, solar, start_hour=-1, end_hour=-1, start_battery_perc=0.050504167)
        processed_images.append(sum(processing_rates))
        processed_steps += len(processing_rates)

    print("Processed sum",sum(processed_images))
    print("Processed avg per day",mean(processed_images))
    print("Processed steps",processed_steps)
    exit(0)
    """
    def check_optimal(processing_slots:list[float], start_day:int, flip_index:int = -1):
        start_day = start_day*24*60*60
        def E_s(t):
            return solar.get_solar_w(start_day + DELTA_T + t * DELTA_T) * DELTA_T
        
        # Take first
        if True:
            if flip_index > 0 and len(processing_slots) < flip_index:
                processing_slots[i] = MAX_PR#not processing_slots[i]
            else:
                for i in range(len(processing_slots)):
                    if processing_slots[i]<MAX_PR:
                        processing_slots[i] += 1
                        break
        else: # Take random
            all_negatives = list(filter(lambda x: not x[1],enumerate(processing_slots)))
            random_false_index = all_negatives[randint(0,len(all_negatives)-1)][0]
            processing_slots[random_false_index] = not processing_slots[random_false_index]
        
        battery_j = MAX_BATTERY_J*0.3
        negative_battery = Falfind_optimalse
        for t in range(1,len(processing_slots)+1):
            battery_j += E_s(t) - E_IDLE_J - processing_slots[t-1] * E_PROCESSING_FOR_IMG_J
            battery_j = min(battery_j, MAX_BATTERY_J)
            if battery_j < 0:
                negative_battery = True
                #print("Negative battery", battery_j)
        print("Final battery",battery_j)
        print("Processed images",sum(processing_slots))
        return negative_battery


    l_index = enumerate(l)
    l_index = list(filter(lambda x:x[1]<MAX_PR, l_index))
    idle_indexes = list(map(lambda x:x[0], l_index))
    print("Idle indexes",idle_indexes)
    violations = 0
    for idle_index in idle_indexes:
        local_list = l.copy()
        is_negative = check_optimal(local_list,1, flip_index=idle_index)
        print("Tryng flipping index",idle_index,"from",local_list[idle_index],"to",MAX_PR)
        if not is_negative:
            print("Error!")
            break
        else:
            violations += 1
    print("Negatives indexes len",len(idle_indexes), "Violations",violations)
    """
