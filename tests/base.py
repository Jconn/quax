import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, QConv, Dequantize, QModule
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB


def run_model_vs_tflite(model, input_data, act_bits, use_quantize, params=None):

    if params is None:
        # hack - assume init has happened if params are passed
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, input_data)


    # Apply model
    output = model.apply(params, input_data)

    # Convert to TFLite
    converter = FBB()
    tflite_model = converter.convert(model, params, x=input_data)

    # Use io.BytesIO to simulate a file in memory
    tflite_model_file = io.BytesIO(tflite_model)
    with open("tflite_model.tflite", "wb") as f:
        f.write(tflite_model)

    # Set up TFLite interpreter using the RAM file
    interpreter = tf.lite.Interpreter(model_content=tflite_model_file.read())
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    dtype = bits_to_type(act_bits)
    # Quantize the input
    if not use_quantize:
        input_scale, input_zero_point = input_details[0]['quantization']
        quantized_input = np.round(input_data / input_scale + input_zero_point).astype(dtype)
        input_data = quantized_input

    # Run inference on TFLite model
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_index)

    # Dequantize the TFLite output
    if not use_quantize:
        output_scale, output_zero_point = output_details[0]['quantization']
        tflite_output = (tflite_output.astype(np.float32) - output_zero_point) * output_scale

    # Compare TFLite and original model outputs
    assert np.allclose(output, tflite_output, atol=1e-2), "Outputs do not match!"


