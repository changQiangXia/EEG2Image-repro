import json
import os

import tensorflow as tf


def configure_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def build_strategy():
    gpus = configure_runtime()
    if not gpus:
        return tf.distribute.OneDeviceStrategy("/CPU:0")
    if len(gpus) == 1:
        return tf.distribute.OneDeviceStrategy("/GPU:0")
    return tf.distribute.MirroredStrategy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
