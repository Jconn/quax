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

@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [8, 16])
@pytest.mark.parametrize("strides", [(1, 1), (1, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 3), (3, 3)])
@pytest.mark.parametrize("input_shape", [(1, 10,10,1), (1,6,6,7), (2,6,6,2)])
@pytest.mark.parametrize("use_quantize", [False, True])

def test_model_vs_tflite(act_bits, weight_bits, features, strides, kernel_size, input_shape,use_quantize):
    # Create a small CNN model
    class CNN(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            x = QConv(features=features, strides=strides, kernel_size=kernel_size,
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID')(x)
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x

    # Initialize model
    rng = jax.random.PRNGKey(0)
    cnn_model = CNN(train_quant=True, use_quantize=use_quantize)

    # Generate random input data
    #input_data = jnp.ones(input_shape)
    input_data = jax.random.uniform(jax.random.key(0),shape=input_shape)
    
    params = cnn_model.init(rng, input_data)

    # Apply model
    output = cnn_model.apply(params, input_data)

    # Convert to TFLite
    converter = FBB()
    tflite_model = converter.convert(cnn_model, params, x=input_data)

    # Use io.BytesIO to simulate a file in memory
    tflite_model_file = io.BytesIO(tflite_model)

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
    assert np.allclose(output, tflite_output, atol=1e-2), f"Outputs do not match! Act Bits: {act_bits}, Weight Bits: {weight_bits}, Features: {features}, Strides: {strides}, Kernel Size: {kernel_size}"


