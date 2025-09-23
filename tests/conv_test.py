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

@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [8, 16])
@pytest.mark.parametrize("strides", [(1, 1), (1, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 3), (3, 3)])
@pytest.mark.parametrize("input_shape", [(1, 10,10,1), (1,300,300,7), (2,6,6,2)])
@pytest.mark.parametrize("use_quantize", [False, True])
@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("padding", ['VALID', 'SAME'])

def test_cnn(act_bits, weight_bits, features, strides, kernel_size, input_shape,use_quantize, use_relu, padding):
    # Create a small CNN model
    class CNN(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            act_fn = nn.relu if use_relu else None 
            x = QConv(features=features, strides=strides, kernel_size=kernel_size,
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding=padding,act_fn=act_fn)(x)

            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x


    # Generate random input data
    #input_data = jnp.ones(input_shape)
    rng = jax.random.PRNGKey(0)
    input_data = jax.random.uniform(rng,shape=input_shape)
    cnn_model = CNN(train_quant=True, use_quantize=use_quantize)
    #TODO - disable the eval model 
    eval_model = CNN(train_quant=False, use_quantize=use_quantize)

    params = cnn_model.init(rng, input_data)
    test_data = jax.random.uniform(rng,shape=input_shape)
    run_model_vs_tflite(eval_model, input_data, act_bits, use_quantize, params=params)
    run_model_vs_tflite(cnn_model, input_data, act_bits, use_quantize, params=params)
    save_and_load_model(cnn_model, params, test_data)

