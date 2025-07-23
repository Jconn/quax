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
from base import run_model_vs_tflite

@pytest.mark.parametrize("act_bits", [8])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [8, 16])
@pytest.mark.parametrize("strides", [(1, 1), (1, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 3), (3, 3)])
@pytest.mark.parametrize("input_shape", [(1, 10,10,1), (1,6,6,7), (2,6,6,2)])
@pytest.mark.parametrize("use_quantize", [False, True])
@pytest.mark.parametrize("use_relu", [False, True])

def test_transpose(act_bits, weight_bits, features, strides, kernel_size, input_shape,use_quantize, use_relu):
    # Create a small CNN model
    class CNNTranspose(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            act_fn = nn.relu if use_relu else None 
            x = QConv(features=features, strides=strides, kernel_size=kernel_size,
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID',act_fn=act_fn)(x)
            x = x.transpose([0,2,1,3])
            x = QConv(features=2, strides=strides, kernel_size=(1,1),
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID',act_fn=act_fn)(x)
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x


    # Generate random input data
    #input_data = jnp.ones(input_shape)
    input_data = jax.random.uniform(jax.random.key(0),shape=input_shape)
    cnn_model = CNNTranspose(train_quant=True, use_quantize=use_quantize)
    #TODO - disable the eval model 
    eval_model = CNNTranspose(train_quant=False, use_quantize=use_quantize)

    rng = jax.random.PRNGKey(0)
    params = cnn_model.init(rng, input_data)
    run_model_vs_tflite(eval_model, input_data, act_bits, use_quantize, params=params)
    run_model_vs_tflite(cnn_model, input_data, act_bits, use_quantize, params=params)

