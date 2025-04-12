import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="models/autoencoder.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input type:", input_details[0]['dtype'])
print("Output type:", output_details[0]['dtype'])