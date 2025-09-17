import csv
from datetime import datetime
from datetime import timedelta


class Solar:
    def __init__(self, csv_path: str, scale_factor: float = 1, max_power:int = 0, enable_cache = False):
        self.values = []
        self.max_power_w = max_power
        self.enable_cache = enable_cache
        self.cache = {}
        self.day_cache = {}
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            # print(list(zip(range(0,len(header)),header)))
            for line in reader:
                dt = datetime.fromisoformat(line[26])
                start_of_the_year = dt.replace(
                    month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                delta = dt - start_of_the_year
                info = {
                    'cloud_opacity':line[7],
                    'humidty':line[15],
                    'pressure':line[16],
                }
                
                value = (delta.total_seconds(), int(line[12])*scale_factor, dt, info)
                self.values.append(value)

    def get_solar_w(self, time_s: int):
        # print("Solar time:", time_s)
        if self.enable_cache and str(time_s) in self.cache.keys():
            return self.cache[str(time_s)]
        for i in range(len(self.values)):
            if self.values[i][0] == time_s:
                v = self.values[i][1]
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                if self.enable_cache:
                    self.cache[str(time_s)]=v
                return v
            elif i+1 < len(self.values) and self.values[i][0] < time_s and self.values[i+1][0] > time_s:
                # print(f"Found: {self.values[i]}")
                current_t = self.values[i][0]
                next_t = self.values[i+1][0]
                delta_second = next_t - current_t
                delta_value = self.values[i+1][1]-self.values[i][1]
                fraction = delta_value / delta_second
                v = self.values[i][1]+fraction*(time_s-current_t)
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                if self.enable_cache:
                    self.cache[str(time_s)]=v
                return v
        # print("Solar not found")
        return -1
    
    def get_info(self, time_s: int, field:str):
        # print("Solar time:", time_s)
        if self.enable_cache and str(time_s) in self.cache.keys():
            return self.cache[str(time_s)]
        
        for i in range(len(self.values)):
            if self.values[i][0] == time_s:
                v = self.values[i][3][field]
                # if self.enable_cache:
                #     self.cache[str(time_s)]=v
                return v
            elif i+1 < len(self.values) and self.values[i][0] < time_s and self.values[i+1][0] > time_s:
                # print(f"Found: {self.values[i]}")
                current_t = self.values[i][0]
                next_t = self.values[i+1][0]
                delta_second = next_t - current_t
                delta_value = self.values[i+1][3][field]-self.values[i][3][field]
                fraction = delta_value / delta_second
                v = self.values[i][3][field]+fraction*(time_s-current_t)
                v = min(self.max_power_w,v) if self.max_power_w > 0 else v
                # if self.enable_cache:
                #     self.cache[str(time_s)]=v
                return v
        # print("Solar not found")
        return -1

    def get_datetime(self, time_s: int) -> datetime:
        for i in range(len(self.values)):
            if self.values[i][0] == time_s:
                return self.values[i][2]
            elif (i+1 < len(self.values) and self.values[i][0] < time_s and self.values[i+1][0] > time_s) or (i+1 == len(self.values) and self.values[i][0] < time_s):
                # print(f"Found: {self.values[i]}")
                prev_s = self.values[i][0]
                prev_date: datetime = self.values[i][2]
                return prev_date+timedelta(seconds=time_s-prev_s)
        # print("Solar not found")
        return None

    def get_next_sunrise(self, time_s: int) -> int:
        # Step 1: Find next item in the data
        # Step 2: Find the sunrise
        for i in range(len(self.values)):
            if self.values[i][0] >= time_s:
                # next_item_found
                # Go head untill is night
                k = i
                while self.values[k][1] == 0:
                    k += 1
                return self.values[k][0]


    # Current step if the last with the night
    def is_sunrise(self, time_s:int, step_size_s:int, steps:int = 5):
        if self.get_solar_w(time_s) != 0: #Now i dark
            return False
        
        for i in range(steps-1):
            next_time = time_s + step_size_s*(i+1) #Next steps with light
            if self.get_solar_w(next_time) <= 0:
                return False
        return True
    
    def is_sunset(self, time_s:int, step_size_s:int, steps:int = 5):
        if self.get_solar_w(time_s) == 0: #If now there is light
            return False
        
        for i in range(steps-1): #Next steps witout light
            next_time = time_s + step_size_s*(i+1)
            if self.get_solar_w(next_time) > 0:
                return False
        return True
    
    def are_steps_with_at_least(self, time_s:int, step_size_s:int, steps:int, power_w:int) -> bool:
        for i in range(steps):
            next_time = time_s + step_size_s*i
            if self.get_solar_w(next_time) < power_w:
                return False
        return True
    
    def is_night(self, time_s:int, step_size_s:int, steps:int) -> bool:
        for i in range(steps):
            next_time = time_s + step_size_s*i
            if self.get_solar_w(next_time) > 0:
                return False
        return True
    
    def solar_slope(self, time_s:int, future_m:int) -> bool:
        a = self.get_solar_w(time_s)
        b = self.get_solar_w(time_s+future_m)
        if a < b:
            return 1
        elif a > b:
            return -1
        else: #a > b
            return 0
        
    def get_day_avg(self, day:int) -> float:
        if str(day) in self.day_cache.keys():
            return self.day_cache[str(day)]
        acc = 0
        #counter = 0
        for x in self.values:
            if x[2].timetuple().tm_yday == day:
                acc += x[1]
                #counter += 1
        self.day_cache[str(day)]=acc
        return acc
    
    def get_future_prediction_w(self, time_s:int, step_size_t:int, interval_m:int) -> float:
        accumulated_power_w = 0
        for t in range(time_s, time_s+interval_m*60,step_size_t):
            e = self.get_solar_w(t)
            if e < 0:
                break
            accumulated_power_w += e
        return accumulated_power_w


