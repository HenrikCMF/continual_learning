import csv
from ES import ES_station
screen="bandwidth"#energy|bandwidth
results_file = "pump_no_IoTadapt.csv"
#with open(results_file, 'w', newline='') as f:
#    csv.writer(f).writerow(['bandwidth_kbps', 'TP', 'FP', 'avg_throughput_kbps'])
if screen=="energy":
    bandwidths = 1000
    energy_budget = [28, 85,142,199,255,312]

    for e in energy_budget:
        es = ES_station("received", bandwidth=bandwidths, energy_budget=e)
        TP, FP, avg_throughput = es.run()
        with open(results_file, 'a', newline='') as f:
            csv.writer(f).writerow([e, TP, FP, avg_throughput])
        print(f"[run_ES] bw={bandwidths} kbps done — TP={TP}, FP={FP}, avg_throughput={avg_throughput:.1f} kbps")

elif screen=="bandwidth":
    bandwidths = [100,300,500,700,900,1100]
    energy_budget = 284
    for b in bandwidths:
        es = ES_station("received", bandwidth=b, energy_budget=energy_budget)
        TP, FP, avg_throughput = es.run()
        with open(results_file, 'a', newline='') as f:
            csv.writer(f).writerow([b, TP, FP, avg_throughput])
        print(f"[run_ES] bw={bandwidths} kbps done — TP={TP}, FP={FP}, avg_throughput={avg_throughput:.1f} kbps")