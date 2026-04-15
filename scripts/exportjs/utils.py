import tensorflow as tf
import numpy as np
from NN.models.npz_utils import model_to_raw_dict


def stabilize(model):
    weights = model_to_raw_dict(model)
    for w in weights.values():
        w.assign(tf.where(0.0 < w, 0.01, -0.01))


def stabilize_input(input_data: np.array):
    return np.where(0.0 < input_data, 0.01, -0.01)
