import pulp as pl
from lib.solar import solar
from max_images import save_plot
from random import randint

PANEL_AREA_M2 = 0.55 * 0.51  # m2
EFFICIENCY = 0.1426
MAX_POWER_W = 40  # W
solar = solar.Solar(
    "../solcast2025.csv",
    scale_factor=PANEL_AREA_M2 * EFFICIENCY,
    max_power=MAX_POWER_W,
    enable_cache=True,
)

MAX_BATTERY_J = 6.6 * 3.7 * 3600
DELTA_T = 5 * 60
E_IDLE_J = 5 * 0.05 * DELTA_T
E_PROCESSING_J = (5 * DELTA_T) - E_IDLE_J

def find_optimal(start_day_:int):
    # Parametri

    T_max = (1*24*60)//5#34560
    start_day = start_day_*24*60*60
    M = 10 * MAX_BATTERY_J  # big-M, deve essere molto più grande della batteria

    def E_s(t):
        return solar.get_solar_w(start_day + DELTA_T + t * DELTA_T) * DELTA_T

    # Problema
    prob = pl.LpProblem("myProblem", pl.LpMaximize)

    # Variabili decisionali
    x = pl.LpVariable.dicts("x", range(1, T_max), cat="Binary")
    E = pl.LpVariable.dicts("E", range(T_max), lowBound=0, upBound=MAX_BATTERY_J, cat="Continuous")
    y = pl.LpVariable.dicts("y", range(1, T_max), cat="Continuous")   # livello teorico prima del min
    z = pl.LpVariable.dicts("z", range(1, T_max), cat="Binary")       # decide se saturare o no

    # Condizione iniziale
    prob += E[0] == MAX_BATTERY_J * 0.5, "Condizione_iniziale"

    # Vincoli ricorsivi con min linearizzato
    for t in range(1, T_max):
        # livello teorico senza taglio
        prob += y[t] == E[t - 1] + E_s(t) - E_IDLE_J - E_PROCESSING_J * x[t], f"Def_y_{t}"

        # vincoli per E[t] = min(y[t], max_battery_j)
        prob += E[t] <= y[t], f"Min1_{t}"
        prob += E[t] <= MAX_BATTERY_J, f"Min2_{t}"
        prob += E[t] >= y[t] - M * (1 - z[t]), f"Min3_{t}"
        prob += E[t] >= MAX_BATTERY_J - M * z[t], f"Min4_{t}"

    # Funzione obiettivo (massimizza la somma degli x_t)
    prob += pl.lpSum(x[t] for t in range(1, T_max)), "Somma_X"

    # Risolvi
    prob.solve()

    # Risultati
    print("Stato:", pl.LpStatus[prob.status])
    print("Final battery", E[T_max - 1].value())
    # for t in range(1, T_max):
    #     print(f"x[{t}] = {x[t].value()},   E[{t}] = {E[t].value()},   y[{t}] = {y[t].value()},   z[{t}] = {z[t].value()}")

    battery_levels = [1 if z[t].value()==0 else E[t].value()/MAX_BATTERY_J for t in range(1,T_max)]
    # save_plot(
    #     battery_levels=battery_levels,
    #     recharges=[E_s(t)/(MAX_POWER_W*DELTA_T) for t in range(1,T_max)],
    #     consumptions=[x[t].value() for t in range(1,T_max)],
    #     times=[],
    #     sunrises=[],
    #     path=f"pulp/max_images_pulp_{start_day_}.png"
    # )
    return sum([x[t].value() for t in range(1,T_max)]), [bool(x[t].value()) for t in range(1,T_max)]


TEST_ALL_DAYS = False
if TEST_ALL_DAYS:
    processed_images = []
    for d in range(0,120,1):
        p = find_optimal(d)
        processed_images.append(int(p))
    print(processed_images)
else:
    p, l = find_optimal(6)
    print(f"Processed images {p}")

def check_optimal(processing_slots:list[bool], start_day:int, flip_index:int = -1):
    start_day = start_day*24*60*60
    def E_s(t):
        return solar.get_solar_w(start_day + DELTA_T + t * DELTA_T) * DELTA_T
    
    # Take first
    if True:
        if flip_index > 0 and len(processing_slots) < flip_index:
            processing_slots[i] = not processing_slots[i]
        else:
            for i in range(len(processing_slots)):
                if not processing_slots[i]:
                    processing_slots[i] = True
                    break
    else: # Take random
        all_negatives = list(filter(lambda x: not x[1],enumerate(processing_slots)))
        random_false_index = all_negatives[randint(0,len(all_negatives)-1)][0]
        processing_slots[random_false_index] = not processing_slots[random_false_index]
    
    battery_j = MAX_BATTERY_J*0.5
    negative_battery = False
    for t in range(1,len(processing_slots)+1):
        battery_j += E_s(t) - E_IDLE_J - processing_slots[t-1] * E_PROCESSING_J
        battery_j = min(battery_j, MAX_BATTERY_J)
        if battery_j < 0:
            negative_battery = True
            print("Negative battery", battery_j)
    print("Final battery",battery_j)
    print("Processed images",sum(processing_slots))
    return negative_battery

l_index = enumerate(l)
l_index = list(filter(lambda x:not x[1], l_index))
idle_indexes = list(map(lambda x:x[0], l_index))
print("Idle indexes",idle_indexes)
violations = 0
for idle_index in idle_indexes:
    local_list = l.copy()
    is_negative = check_optimal(local_list,6, flip_index=idle_index)
    if not is_negative:
        print("Error!")
    else:
        violations += 1
print("Negatives indexes len",len(idle_indexes), "Violations",violations)
exit(0)