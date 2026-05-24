import csv
from IoT_device import iot_device
import time
bandwidths = [900,1100]
energy_budget = 284
results_file = "SML_IoT.csv"

#with open(results_file, 'w', newline='') as f:
#    csv.writer(f).writerow([
#        'bandwidth_kbps', 'total_energy', 'time_transmitting_s', 'time_receiving_s',
#        'total_sent_kb', 'total_received_kb', 'num_inferences', 'avg_throughput_kbps'])

for bw in bandwidths:
    time.sleep(10)
    device = iot_device("received", bandwidth=bw, energy_budget=energy_budget)
    time_tx, time_rx, sent, received, inferences, avg_tp, total_energy = device.run()
    with open(results_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            bw, total_energy, time_tx, time_rx,
            sent / 1024, received / 1024, inferences, avg_tp
        ])
    print(f"[run_IoT] bw={bw} kbps done — energy={total_energy:.2f}, inferences={inferences}")
