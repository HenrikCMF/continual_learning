from Base import Base_station
import numpy as np
import time
import csv
import os
file_path = "plots/MLP_results.csv"

if not os.path.isfile(file_path):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "TP", "FP"])

start = 0.08
stop = 0.99
step = 0.08
num_steps = int((stop - start) / step) + 1
for idx in range(num_steps):
    i = round(start + step * idx, 2)
    time.sleep(10)
    bs=Base_station("received", i)
    TP, FP = bs.run(i)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([i, TP, FP])