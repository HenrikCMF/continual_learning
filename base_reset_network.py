import subprocess
import json

#Just used to reset my network again, in case a test fails
with open("configs.json", "r") as file:
        configs = json.load(file)
interface=configs['baseNET_INTERFACE']
subprocess.run(f"sudo tc qdisc del dev {interface} root", shell=True)