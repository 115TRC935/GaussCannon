import numpy as np

epsilon = 1e-10

def safe_log(x):
    return np.log(np.maximum(x, epsilon))

def safe_sqrt(x):
    return np.sqrt(np.maximum(x, epsilon))

def safe_divide(x, y):
    return np.divide(x, np.maximum(np.abs(y), epsilon))

def safe_int(x):
    return np.int32(np.maximum(x, epsilon))

def safe_float(x):
    return np.float32(np.maximum(x, epsilon))

def safe_exp(x):
    return np.exp(np.minimum(x, 700))  # Evitar overflow en exp

def safe_pow(x, y):
    return np.power(np.maximum(x, epsilon), y)

def safe_sum(x, axis=None):
    return np.sum(np.maximum(x, epsilon), axis=axis)

def safe_mean(x, axis=None):
    return np.mean(np.maximum(x, epsilon), axis=axis)

def safe_max(x, axis=None):
    return np.max(np.maximum(x, epsilon), axis=axis)

def safe_min(x, axis=None):
    return np.min(np.maximum(x, epsilon), axis=axis)

def safe_clip(x, min_val, max_val):
    return np.clip(x, min_val, max_val)

def safe_concatenate(arrays, axis=0):
    return np.concatenate([np.maximum(arr, epsilon) for arr in arrays], axis=axis)

def safe_stack(arrays, axis=0):
    return np.stack([np.maximum(arr, epsilon) for arr in arrays], axis=axis)

def safe_reshape(arr, newshape):
    return np.reshape(np.maximum(arr, epsilon), newshape)

def safe_arctan(x):
    return np.arctan(np.maximum(x, epsilon))

def safe_sin(x):
    return np.sin(np.maximum(x, epsilon))

def safe_cos(x):
    return np.cos(np.maximum(x, epsilon))