import numpy as np
def bits_to_type(bits):
    if bits <= 8:
        dtype = np.int8
    elif bits <= 16:
        dtype = np.int16
    elif bits <= 32:
        dtype = np.int32
    elif bits <= 64:
        dtype = np.int64
    return dtype

