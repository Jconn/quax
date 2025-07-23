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
from base import run_model_vs_tflite, save_and_load_model

@pytest.mark.parametrize("act_bits", [8])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [8])
@pytest.mark.parametrize("strides", [(1, 1)])
@pytest.mark.parametrize("kernel_size", [(1, 1)])
@pytest.mark.parametrize("input_shape", [(3,6,6,7)])
@pytest.mark.parametrize("slice_index", [0,1,2,3,4])
@pytest.mark.parametrize("use_quantize", [False, True])
@pytest.mark.parametrize("use_relu", [False])

def test_slice(act_bits, weight_bits, features, strides, kernel_size, input_shape,slice_index, use_quantize, use_relu):
    # Create a small CNN model
    class FCSlice(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            act_fn = nn.relu if use_relu else None 
            x = QConv(features=features, strides=strides, kernel_size=kernel_size,
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID',act_fn=act_fn)(x)
            slice_x = x[1,..., slice_index]

            x = Dequantize(to_tflite=self.use_quantize)(slice_x)
            return x

    fc_model = FCSlice(train_quant=True, use_quantize=use_quantize)

    # Generate random input data
    #input_data = jnp.ones(input_shape)
    input_data = jax.random.uniform(jax.random.key(0),shape=input_shape)

    rng = jax.random.PRNGKey(0)
    input_data = jax.random.uniform(rng,shape=input_shape)

    params = fc_model.init(rng, input_data)
    test_data = jax.random.uniform(rng,shape=input_shape)

    run_model_vs_tflite(fc_model, input_data, act_bits, use_quantize)
    save_and_load_model(fc_model, params, test_data)

