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
import orbax.checkpoint as ocp
import tempfile
from pathlib import Path
from quax.tflite_numerics import tflite_round
#TODO - fix interpreter
#from ai_edge_litert.interpreter import Interpreter
def run_model_vs_tflite(model, input_data, act_bits, use_quantize, params=None, tolerance = None):

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
    with open("debug_model.tflite", "wb") as f:
        f.write(tflite_model)

    # Set up TFLite interpreter using the RAM file
    interpreter_ref = tf.lite.Interpreter(model_content=tflite_model_file.read(),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    #interpreter = tf.lite.Interpreter(model_content=tflite_model_file.read()) 
    interpreter = interpreter_ref

    
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    dtype = bits_to_type(act_bits)
    # Quantize the input
    float_input = input_data

    if not use_quantize:
        input_scale, input_zero_point = input_details[0]['quantization']
        # must quantize the same way
        quantized_input = input_data / input_scale + input_zero_point 
        quantized_input =  tflite_round(quantized_input).astype(dtype)
        input_data = quantized_input

    # Run inference on TFLite model
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_index)
    orig_output = tflite_output

    # Dequantize the TFLite output
    if not use_quantize:
        output_scale, output_zero_point = output_details[0]['quantization']
        tflite_output = (tflite_output.astype(np.float32) - output_zero_point) * output_scale
    else:
        output_scale = params['quax']['Dequantize_0']['input'].scale
        output_zero_point = params['quax']['Dequantize_0']['input'].zero_point

    # Compare TFLite and original model outputs

    # TODO - slice for some reason produces incorrect shape, even though tflite model shape is correct 
    #        for now, just try a squeeze as a stopgap
    if tflite_output.shape != output.shape:
        tflite_output = tflite_output.squeeze(-1)
    
    def requantize(x):
        return tflite_round(x / output_scale + output_zero_point)
    req_output = requantize(output)
    req_tflite_output = requantize(tflite_output)
    diff = jnp.abs(req_output - req_tflite_output)


    if tolerance is None:
        tolerance = 1.05
    if diff.max() > tolerance:
        diff_sums = (diff != 0).sum()
        size = len(diff.reshape(-1))
        percent_diff = round(diff_sums / size, 2)
        raise ValueError(f"Outputs do not match! Max diff: {diff.max()}, Num diff: {diff_sums}, {percent_diff}")



def save_and_load_model(model, save_params, input_data):
    checkpointer = ocp.StandardCheckpointer()
    rng = jax.random.PRNGKey(0)
    tmp_params = model.init(rng, input_data)
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, tmp_params)

    with tempfile.TemporaryDirectory() as ckpt_path:
        # tmpdir is a string path you can pass to your API
        ckpt_path = Path(ckpt_path) /  'checkpoint'
        checkpointer.save(ckpt_path, save_params)
        load_params = checkpointer.restore(ckpt_path, target=abstract)
    save_output = model.apply(save_params, input_data) 
    load_output = model.apply(load_params, input_data) 
    assert np.allclose(save_output, load_output, atol=1e-2), "save and load outputs do not match!"

