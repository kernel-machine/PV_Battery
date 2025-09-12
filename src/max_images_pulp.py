import pulp as pl
from lib.solar import solar
from max_images import save_plot

panel_area_m2 = 0.55 * 0.51  # m2
efficiency = 0.1426
max_power_w = 40  # W
solar = solar.Solar(
    "../solcast2025.csv",
    scale_factor=panel_area_m2 * efficiency,
    max_power=max_power_w,
    enable_cache=True,
)

def find_optimal(start_day_:int):
    # Parametri
    max_battery_j = 6.6 * 3.7 * 3600
    delta_t = 5 * 60
    e_idle_j = 5 * 0.05 * delta_t
    e_img_j = (5 * delta_t) - e_idle_j
    T_max = (1*24*60)//5#34560
    start_day = start_day_*24*60*60
    M = 10 * max_battery_j  # big-M, deve essere molto più grande della batteria

    def E_s(t):
        return solar.get_solar_w(start_day + delta_t + t * delta_t) * delta_t

    # Problema
    prob = pl.LpProblem("myProblem", pl.LpMaximize)

    # Variabili decisionali
    x = pl.LpVariable.dicts("x", range(1, T_max), cat="Binary")
    E = pl.LpVariable.dicts("E", range(T_max), lowBound=0, upBound=max_battery_j, cat="Continuous")
    y = pl.LpVariable.dicts("y", range(1, T_max), cat="Continuous")   # livello teorico prima del min
    z = pl.LpVariable.dicts("z", range(1, T_max), cat="Binary")       # decide se saturare o no

    # Condizione iniziale
    prob += E[0] == max_battery_j * 0.5, "Condizione_iniziale"

    # Vincoli ricorsivi con min linearizzato
    for t in range(1, T_max):
        # livello teorico senza taglio
        prob += y[t] == E[t - 1] + E_s(t) - e_idle_j - e_img_j * x[t], f"Def_y_{t}"

        # vincoli per E[t] = min(y[t], max_battery_j)
        prob += E[t] <= y[t], f"Min1_{t}"
        prob += E[t] <= max_battery_j, f"Min2_{t}"
        prob += E[t] >= y[t] - M * (1 - z[t]), f"Min3_{t}"
        prob += E[t] >= max_battery_j - M * z[t], f"Min4_{t}"

    # Funzione obiettivo (massimizza la somma degli x_t)
    prob += pl.lpSum(x[t] for t in range(1, T_max)), "Somma_X"

    # Risolvi
    prob.solve()

    # Risultati
    print("Stato:", pl.LpStatus[prob.status])
    print("Final battery", E[T_max - 1].value())
    # for t in range(1, T_max):
    #     print(f"x[{t}] = {x[t].value()},   E[{t}] = {E[t].value()},   y[{t}] = {y[t].value()},   z[{t}] = {z[t].value()}")

    battery_levels = [1 if z[t].value()==0 else E[t].value()/max_battery_j for t in range(1,T_max)]
    save_plot(
        battery_levels=battery_levels,
        recharges=[E_s(t)/(max_power_w*delta_t) for t in range(1,T_max)],
        consumptions=[x[t].value() for t in range(1,T_max)],
        times=[],
        sunrises=[],
        path=f"pulp/max_images_pulp_{start_day_}.png"
    )
    return sum([x[t].value() for t in range(1,T_max)])


processed_images = []
for w in range(0,120,1):
    p = find_optimal(w)
    processed_images.append(int(p))
print(processed_images)