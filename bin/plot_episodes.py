import csv
import matplotlib.pyplot as plt

with open('episodes.csv', 'r') as f:
    episodes = [int(row[0]) for row in csv.reader(f) if row]

first_one = next((i for i, e in enumerate(episodes) if e == 1), None)
if first_one is not None:
    episodes = episodes[first_one + 1:]

window = 10
x, y = [], []
for i in range(0, len(episodes) - window + 1):#, window):
    chunk = episodes[i:i + window]
    x.append(i + window)
    y.append(chunk.count(0))

plt.figure()
plt.plot(x, y, marker='o')
plt.xlabel('Episode')
plt.ylabel('False positives (0s) per 10 episodes')
plt.title('False positives over time')
plt.tight_layout()
plt.savefig('episodes_plot.png')
plt.show()
