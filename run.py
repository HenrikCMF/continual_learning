import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric")
from Base import Base_station
from Edge import edge_device
import numpy as np
import time
import csv
import os
import queue
import threading


if __name__=="__main__":
    i=300
    Base_queue=queue.Queue()
    Edge_queue=queue.Queue()
    result_Q=queue.Queue()
    bs=Base_station("received", i, Base_queue, Edge_queue)
    #TP, FP, throughput = bs.run(i)
  
    bt=threading.Thread(target=bs.run, args=(i,result_Q))
    bt.start()
    #time.sleep(10)
    edge=edge_device("received", i, Edge_queue, Base_queue)
    et=threading.Thread(target=edge.run, args=(i,result_Q))
    et.start()
    bt.join()
    et.join()
    results={}
    while not result_Q.empty():
        results.update(result_Q.get())
    print(results)
        