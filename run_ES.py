import csv
from ES import ES_station

bandwidths = 1000
energy_budget = [47,94,142,189,236,284]
results_file = "dhawk_energy_ES_2.csv"

#with open(results_file, 'w', newline='') as f:
#    csv.writer(f).writerow(['bandwidth_kbps', 'TP', 'FP', 'avg_throughput_kbps'])

for e in energy_budget:
    es = ES_station("received", bandwidth=bandwidths, energy_budget=e)
    TP, FP, avg_throughput = es.run()
    with open(results_file, 'a', newline='') as f:
        csv.writer(f).writerow([e, TP, FP, avg_throughput])
    print(f"[run_ES] bw={bandwidths} kbps done — TP={TP}, FP={FP}, avg_throughput={avg_throughput:.1f} kbps")
