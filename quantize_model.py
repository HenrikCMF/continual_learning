import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import LastValueQuantizer, MovingAverageQuantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer, quantize_apply
from tensorflow_model_optimization.quantization.keras import quantize_apply, quantize_scope
import tensorflow_model_optimization as tfmot
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

class CustomLayerQuantizeConfig(QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [
        (layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
        (layer.bias,   LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
    ]

    def get_activations_and_quantizers(self, layer):
        #return []
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return [MovingAverageQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)]
        #return []

    def get_config(self):
        return {}

def quantize_8_bit(model, example_data, location):
    def representative_data_gen():
        x=example_data
        for i in range(len(x)):
            yield [x[i:i+1].astype(np.float32)]

    quantconverter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantconverter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantconverter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quantconverter.representative_dataset = representative_data_gen
    quantlite=quantconverter.convert()
    with open(location+".tflite", "wb") as f:
        f.write(quantlite)

def get_pruning_wrapper(model, sparsity, epochs, batch_size, num_training_samples):
    total_steps=int(num_training_samples/batch_size*epochs)
    start_step=int(total_steps/2)
    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,    # start training with 0% sparsity
        final_sparsity=sparsity,     # end training with 50% sparsity
        begin_step=start_step,            # when to start pruning
        end_step=total_steps        # when to end pruning
    )
    }
    model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    return model, callbacks

def strip_prune(model):
    return tfmot.sparsity.keras.strip_pruning(model)