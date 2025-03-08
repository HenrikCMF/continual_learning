import subprocess
import json
class network_control():
    def __init__(self, device_type):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        if device_type=='bs':
            self.interface=configs['baseNET_INTERFACE']
        else:
            self.interface=configs['edgeNET_INTERFACE']
        #self.rate_kbps=configs['bandwidth_limit_kbps']
        #self.burst_kbps=configs['burst_limit_kbps']
        #self.latency_ms=configs['buffering_latency_ms']
        #self.packet_loss_pct=configs['packet_loss_pct']
        #self.delay_ms=configs['base_delay_ms']
        #self.jitter_ms=configs['jitter_ms']


    def set_network_conditions(self, rate, burst, latency=None, p_loss=None, delay=None, jitter=None):
        """
        Configures network conditions on a given interface using tc.

        Parameters:
        - interface (str): Network interface (e.g., "eth0", "wlan0").
        - rate_kbps (int): Bandwidth limit in Kbps.
        - burst_kbps (int): Allowed burst size in Kbps.
        - latency_ms (int): Latency before dropping packets in TBF.
        - packet_loss_pct (float): Packet loss percentage (e.g., 5 for 5% loss).
        - delay_ms (int): Mean delay added by netem.
        - jitter_ms (int): Jitter variation for the delay.
        # Example usage
        interface = "eth0"  # Change to your network interface (e.g., "wlan0" for WiFi)
        rate_kbps = 8  # 1 KB/s (8 Kbps)
        burst_kbps = 16  # 2 KB burst
        latency_ms = 400  # Buffering latency in TBF
        packet_loss_pct = 5  # 5% packet loss
        delay_ms = 250  # Base delay
        jitter_ms = 50  # Jitter
        """

        # Convert Kbps to bits per second (tc requires bits)
        rate_bits = (rate * 1000) // 8
        burst_bytes = (burst * 1000) // 8
        # Remove any existing rules
        subprocess.run(f"sudo tc qdisc del dev {self.interface} root", shell=True, stderr=subprocess.DEVNULL)
        print(1)
        # Apply TBF (Token Bucket Filter) for bandwidth limitation
        tbf_command=f"sudo tc qdisc add dev {self.interface} root handle 1: tbf rate {rate_bits}bit burst {burst_bytes}"
    
        # Only add latency to TBF if specified
        if latency is not None:
            tbf_command += f" latency {latency}ms"
        subprocess.run(tbf_command, shell=True)
        # Apply Netem for delay and packet loss under TBF
        if p_loss or delay:
            netem_command = f"sudo tc qdisc add dev {self.interface} parent 1:1 handle 10: netem"
            
            # Add packet loss if specified
            if p_loss is not None:
                netem_command += f" loss {p_loss}%"
            
            # Add delay and jitter if specified
            if delay is not None:
                netem_command += f" delay {delay}ms"
                if jitter is not None:
                    netem_command += f" {jitter}ms"
        
            subprocess.run(netem_command, shell=True)

    def reset_network_conditions(self):
        """Removes all traffic control settings from the specified network interface."""
        subprocess.run(f"sudo tc qdisc del dev {self.interface} root", shell=True)
        print(f"Reset network conditions on {self.interface}.")
    

obj=network_control(device_type='bs')
#obj.set_network_conditions()
obj.reset_network_conditions()