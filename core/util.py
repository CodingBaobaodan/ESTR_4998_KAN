import hashlib
import importlib
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import sys


def cal_conf_hash(config, useless_key=None, hash_len=10):
    if useless_key is None:
        useless_key = ['save_root', 'data_root', 'seed', 'ckpt_path', 'conf_hash', 'use_wandb']

    conf_str = ''
    for k, v in config.items():
        if k not in useless_key:
            conf_str += str(v)

    md5 = hashlib.md5()
    md5.update(conf_str.encode('utf-8'))
    return md5.hexdigest()[:hash_len]


def load_module_from_path(module_name, exp_conf_path):
    spec = importlib.util.spec_from_file_location(module_name, exp_conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Plot the data, red dot means wrong direction prediction.
def plot_confidence_vs_loss(confidences, custom_losses, predictions_tomorrow, true_prices_tomorrow, true_prices_today):
    # Convert lists to NumPy arrays for element-wise operations
    predictions_tomorrow = np.array(predictions_tomorrow)
    true_prices_tomorrow = np.array(true_prices_tomorrow)
    true_prices_today = np.array(true_prices_today)

    # Calculate the penalty mask
    predicted_diff = predictions_tomorrow - true_prices_today
    actual_diff = true_prices_tomorrow - true_prices_today
    penalty_mask = (predicted_diff * actual_diff < 0).astype(float)  # 1 if wrong direction, 0 otherwise

    plt.figure(figsize=(10, 6))
    for i in range(len(confidences)):
        color = 'red' if penalty_mask[i] == 1 else 'blue'
        plt.scatter(confidences[i], custom_losses[i], color=color, alpha=0.5)

    plt.title('Confidence Score vs. Loss')
    plt.xlabel('Confidence Score')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.show()