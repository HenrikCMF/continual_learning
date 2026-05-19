import csv
from ES import ES_station

bandwidths = [500]
energy_budget = 220
results_file = "es_results.csv"

#with open(results_file, 'w', newline='') as f:
#    csv.writer(f).writerow(['bandwidth_kbps', 'TP', 'FP', 'avg_throughput_kbps'])

for bw in bandwidths:
    es = ES_station("received", bandwidth=bw, energy_budget=energy_budget)
    TP, FP, avg_throughput = es.run()
    with open(results_file, 'a', newline='') as f:
        csv.writer(f).writerow([bw, TP, FP, avg_throughput])
    print(f"[run_ES] bw={bw} kbps done — TP={TP}, FP={FP}, avg_throughput={avg_throughput:.1f} kbps")
