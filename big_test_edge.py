from Edge import edge_device
import numpy as np
import time
import csv
import os
import tensorflow as tf
file_path = "plots/ROC_400_N.csv"

#Function for runnning the full test run multiple times but with a changing input for each run
def analyze_model_energy_params(model_path):
    """
    Load a Keras model from file and calculate:
    - N_s: Total weights + biases
    - N_c: Total MAC operations per inference
    - A_s: Total activations (output size across all Dense layers)

    Parameters:
        model_path (str): Path to a saved Keras model (.h5 or SavedModel folder)

    Returns:
        dict: Dictionary with N_s, N_c, A_s
    """
    # Load model without compiling (skip training config)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    # Build the model if needed (e.g., h5 might not restore input shape)
    if not model.built:
        # Try to infer input shape from the first layer
        for layer in model.layers:
            if hasattr(layer, 'input_shape') and isinstance(layer.input_shape, tuple):
                dummy_input = tf.random.normal(shape=(1, *layer.input_shape[1:]))
                model(dummy_input)
                break

    N_s = 0  # Total parameters (weights + biases)
    N_c = 0  # Total MACs
    A_s = 0  # Total activations

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if len(weights) >= 1:
                input_units, output_units = weights[0].shape  # shape: (input_dim, output_dim)
                num_weights = input_units * output_units
                num_biases = output_units if len(weights) > 1 else 0

                N_s += num_weights + num_biases
                N_c += num_weights
                A_s += output_units

    return N_s, N_c, A_s

if not os.path.isfile(file_path):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["i","time_transmitting", "time_receiving", "total_sent_data", "total_received_data", "num_inferences", "measthrough", "energy"])

start = 0.1
stop = 1
step = 0.1
num_steps = int((stop - start) / step) + 1
for idx in range(num_steps):
    i = round(start + step * idx, 2)
    time.sleep(10)
    bs=edge_device("received", i)
    time_transmitting, time_receiving, total_sent_data, total_received_data, num_inferences, throughput, energy = bs.run(i)
    #N_s, N_c, A_s = analyze_model_energy_params("models/autoencoder.tflite")
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([i, time_transmitting, time_receiving, total_sent_data, total_received_data, num_inferences, throughput, energy])