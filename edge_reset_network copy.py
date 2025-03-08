import subprocess
import json
with open("configs.json", "r") as file:
        configs = json.load(file)
interface=configs['edgeNET_INTERFACE']
subprocess.run(f"sudo tc qdisc del dev {interface} root", shell=True)