import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def determine_quantization_level(quantconverter, level, example_data):
    def representative_data_gen():
        x=example_data
        print("LEN OF X",len(x))
        for i in range(len(x)):
            yield [x[i:i+1].astype(np.float32)]
    match level:
        case 0 :
            print("not compressing")
            return quantconverter
        case 1:
            print("16bit")
            quantconverter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantconverter.target_spec.supported_types = [tf.float16]
            return quantconverter
        case 2:
            print("8bit")
            quantconverter.optimizations = [tf.lite.Optimize.DEFAULT] 
            quantconverter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            quantconverter.representative_dataset = representative_data_gen
            return quantconverter
        case _:
            return quantconverter

def convert_to_tflite(model, example_data, location, quantize=None):
    quantconverter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantizelevel = 2 if quantize else 0

    quantconverter = determine_quantization_level(quantconverter, quantizelevel , example_data)
    quantlite=quantconverter.convert()
    with open(location+".tflite", "wb") as f:
        f.write(quantlite)




def get_pruning_wrapper(model, sparsity, epochs, batch_size, num_training_samples):
    total_steps=int(num_training_samples/batch_size*epochs)
    start_step=int(total_steps/2)
    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,   
        final_sparsity=sparsity,     
        begin_step=start_step,          
        end_step=total_steps    
    )
    }
    model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    return model, callbacks

def strip_prune(model):
    return tfmot.sparsity.keras.strip_pruning(model)