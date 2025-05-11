import numpy as np
class energy():
    def __init__(self):
        self.xi_R = 0.33   # Receive power in Watts
        self.xi_T = 0.79   # Transmit power in Watts
        # Model architecture params (from analyzer or manual estimate)
        self.N_s = 34298       # Total weights + biases
        self.N_c = 33792       # Total MAC operations
        self.A_s = 506         # Total activations
        # Quantization bits
        self.b_q = 32
        self.b_max = 16
        self.E_base = 26.5e-12  # Joules

    def inference_energy(self, quantization_bits):
        b_q=quantization_bits
        E_MUAC = self.E_base * (b_q / self.b_max)**1.25
        E_L = self.E_base * (b_q / self.b_max)
        E_M = 2 * E_L

        # Parallelism factor
        p = 64 * (self.b_max / b_q)

        # E_C: Computation energy (MACs + 3 × activations)
        E_C = E_MUAC * (self.N_c + 3 * self.A_s)

        # E_W: Weight access + partial activations
        E_W = E_M * self.N_s + E_L * (self.N_c / (p**0.5))

        # E_A: Activations + partial compute
        E_A = 2 * E_M * self.A_s + E_L * (self.N_c / (p**0.5))
        E_HW = E_C + E_W + E_A
        return E_HW
    
    def transmission_energy(self, transmission_time):
        tx=np.array(transmission_time)
        E_transmit = self.xi_T * tx
        return E_transmit
    
    def receiving_energy(self, receiving_time):
        rx=np.array(receiving_time)
        E_recv = self.xi_R * rx
        return E_recv

