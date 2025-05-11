from Base import Base_station
import numpy as np
import time
import csv
import os
file_path = "plots/ftest08_server.csv"

if not os.path.isfile(file_path):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "TP", "FP", "size", "measthroughput"])
start = 200
stop = 1000
step = 200
num_steps = int((stop - start) / step) + 1
for idx in range(num_steps):
    i = round(start + step * idx, 2)
    time.sleep(5)
    bs=Base_station("received", i)
    TP, FP, throughput = bs.run(i)
    size = os.path.getsize("models/autoencoder.tflite.zip")
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([i, TP, FP, size, throughput])