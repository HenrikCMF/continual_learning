from Edge import edge_device
import numpy as np
import time
import csv
import os
file_path = "plots/autoencoder_results.csv"

#if not os.path.isfile(file_path):
#    with open(file_path, mode="a", newline="") as file:
#        writer = csv.writer(file)
#        writer.writerow(["i", "TP", "FP"])

start = 0
stop = 200
step = 20
num_steps = int((stop - start) / step) + 1
for idx in range(num_steps):
    i = round(start + step * idx, 2)
    time.sleep(10)
    bs=edge_device("received", 0.2)
    TP, FP = bs.run(i)
    #with open(file_path, mode="a", newline="") as file:
    #    writer = csv.writer(file)
    #    writer.writerow([i, TP, FP])