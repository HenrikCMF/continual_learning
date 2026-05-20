import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
import warnings
from sklearn.exceptions import ConvergenceWarning
from bin.utils import make_initial_data, get_string_config
from bin.IoT_model import IoT_model

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")


def file_size_kb(path):
    return os.path.getsize(path) / 1024


def load_fresh_keras_model(h5_path):
    return tf.keras.models.load_model(h5_path)


def zip_and_measure(tflite_path):
    zip_path = tflite_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(tflite_path, arcname=os.path.basename(tflite_path))
    size = file_size_kb(zip_path)
    os.remove(zip_path)
    return size


def run_experiment(ml_model, h5_path, X_df, sparsity, quantize, models_dir):
    model = load_fresh_keras_model(h5_path)
    if sparsity is not None:
        model = ml_model.manual_prune_weights(model, sparsity)
    temp_name = f"_exp_s{int((sparsity or 0)*100)}_q{int(bool(quantize))}"
    temp_path = os.path.join(models_dir, temp_name)
    ml_model.quantize_model(X_df, model, temp_path, quantize=quantize)
    tflite_path = temp_path + ".tflite"
    size = zip_and_measure(tflite_path)
    os.remove(tflite_path)
    return size


def main():
    config = get_string_config()

    make_initial_data(config['file_paths']['dataset_path'], config['file_paths']['test_files_dir'])
    init_data = os.path.join(
        config['file_paths']['test_files_dir'],
        config['file_paths']['initial_data_file']
    )

    ml_model = IoT_model(init_data, 0.2)
    ml_model.train_initial_model()

    models_dir = config['file_paths']['models_dir']
    h5_path = os.path.join(models_dir, ml_model.model_name + config['file_extensions']['h5_extension'])
    baseline_tflite = os.path.join(models_dir, ml_model.model_name + config['file_extensions']['tflite_extension'])

    X, _ = ml_model.prepare_training_data()
    X_df = pd.DataFrame(X)

    sparsity_levels = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 to 0.9

    results = []

    results.append(("Baseline", "No pruning, no quantization", zip_and_measure(baseline_tflite)))
    results.append(("Baseline", "No pruning, INT8 quantization", run_experiment(ml_model, h5_path, X_df, None, True, models_dir)))

    for s in sparsity_levels:
        results.append(("Pruning only", f"Sparsity {int(s*100)}%", run_experiment(ml_model, h5_path, X_df, s, None, models_dir)))

    for s in sparsity_levels:
        results.append(("Pruning + INT8", f"Sparsity {int(s*100)}% + INT8", run_experiment(ml_model, h5_path, X_df, s, True, models_dir)))

    print("\n\n=== Model Compression Results ===\n")
    current_section = None
    for section, label, size in results:
        if section != current_section:
            print(f"--- {section} ---")
            current_section = section
        print(f"  {label}: {size:.2f} KB")


if __name__ == "__main__":
    main()
