from lib.solar.solar import Solar
import matplotlib.pyplot as plt
import random
import numpy as np
from utils import save_plot
from utils import signal_noise

panel_area_m2 = 0.55*0.51 #m2
efficiency = 0.1426
max_power_w = 40 #W
solar = Solar(csv_path="../solcast2025.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=0.9)
y_values = []
x_values = []
estimate_futures = []
real_futures = []
day = 56
day_offset = day*24*60*60
single_estimates = []

last_random = 1
noised_power = []
start_time = 5*60+day_offset
end_time = 1*24*60*60+day_offset

for i in range(start_time, end_time, 5*60):
    lower_bound_w, upper_bound_w = solar.get_solar_prediction_w(i)
    real_power_w = solar.get_solar_w(i)
    y_values.append((lower_bound_w/max_power_w, real_power_w/max_power_w, upper_bound_w/max_power_w))
    x_values.append(i)
    future_lower_w, future_upper_w = solar.get_estimate_future_prediction_j(i, 5*60, 60)
    future_lower_w /= 60*60*max_power_w
    future_upper_w /= 60*60*max_power_w
    estimate_futures.append((future_lower_w,future_upper_w))

    real_w = solar.get_real_future_prediction_j(i, 5*60, 60)
    real_w /= 60*60*max_power_w
    real_futures.append(real_w)

    single_estimation_w = solar.get_estimate_future_single_prediction_j(i, 5*60, 60)/(60*60*max_power_w)
    single_estimates.append(single_estimation_w)


reals_w = list(map(lambda x:x[1], y_values))
noised_power = signal_noise(reals_w, strength=0.2, max_value=max(reals_w))

#fig, ax = plt.subplots()
#uppers_w = list(map(lambda x:x[2], y_values))
#lowers_w = list(map(lambda x:x[0], y_values))
#uppers_future_w = list(map(lambda x:x[1], estimate_futures))
#lowers_future_w = list(map(lambda x:x[0], estimate_futures))
#avg_estimation = list(map(lambda x:x[0]+(x[1]-x[0])/2, estimate_futures))

#ax.step(x_values,reals_w,color="black", label="Real")
#ax.step(x_values,real_futures,color="orange")
#ax.step(x_values,single_estimates,color="purple")

#ax.step(x_values,noised_power,color="green",label="Forecast")
#ax.fill_between(x_values,uppers_w, lowers_w,alpha=0.2)
#ax.fill_between(x_values,uppers_future_w, lowers_future_w,alpha=0.2, color="green")
first_value = -1
last_value = -1
for i in range(len(reals_w)):
    next_index = min(i+1, len(reals_w)-1)
    if reals_w[i] == 0 and reals_w[next_index] and first_value<0:
        first_value = i
    if last_value < 0 and reals_w[i] > 0 and reals_w[next_index] == 0:
        last_value = i
        break
print(first_value, last_value)
#noised_power = list(filter(lambda x:x>0, noised_power))
reals_w = reals_w[first_value:last_value]
noised_power = noised_power[first_value:last_value]

fields = {
    "Real":reals_w,
    "Forecast":noised_power
}
save_plot(fields,"prova.png")
save_plot(fields,"prova.pdf")
