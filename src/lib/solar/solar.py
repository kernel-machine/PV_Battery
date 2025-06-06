import csv
from datetime import datetime

class Solar:
    def __init__(self, csv_path:str, scale_factor:float=1):
        self.values = []
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            #print(list(zip(range(0,len(header)),header)))
            for line in reader:
                dt = datetime.fromisoformat(line[26])
                start_of_the_year = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                delta = dt - start_of_the_year
                value = (delta.total_seconds(), int(line[12])*scale_factor, line)
                self.values.append(value)

    def get_solar_a(self, time_s:int):
        #print("Solar time:", time_s)
        for i in range(len(self.values)):
            if self.values[i][0]==time_s:
                return self.values[i][1]
            elif i+1 < len(self.values) and self.values[i][0]<time_s and self.values[i+1][0]>time_s:
                #print(f"Found: {self.values[i]}")
                current_t = self.values[i][0]
                next_t = self.values[i+1][0]
                delta_second = next_t - current_t
                delta_value = self.values[i+1][1]-self.values[i][1]
                fraction = delta_value / delta_second
                return self.values[i][1]+fraction*(time_s-current_t)
        #print("Solar not found")
        return -1