import tensorflow as tf
import tensorflow.lite as tflite
interpreter = tf.lite.Interpreter(model_path="models/autoencoder.tflite")
import numpy as np


for detail in interpreter.get_tensor_details():
    print(detail['name'], detail['dtype'])
#interpreter = tf.lite.Interpreter(model_path=location + ".tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input type:", input_details[0]['dtype'])
print("Output type:", output_details[0]['dtype'])
