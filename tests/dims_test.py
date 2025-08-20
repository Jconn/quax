import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, QConv, Dequantize, QModule, expand_dims
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB
from base import run_model_vs_tflite, save_and_load_model

@pytest.mark.parametrize("act_bits", [16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("strides", [(1, 1)])
@pytest.mark.parametrize("kernel_size", [(2, 1)])
@pytest.mark.parametrize("input_shape", [(2,6,2)])
@pytest.mark.parametrize("use_quantize", [True])
@pytest.mark.parametrize("use_relu", [True])

def test_dims(act_bits, weight_bits, strides, kernel_size, input_shape,use_quantize, use_relu):
    # Create a small CNN model
    class CNN(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            x = expand_dims(x,axis=2)
            act_fn = nn.relu if use_relu else None 
            x = QConv(features=1, strides=strides, kernel_size=kernel_size,
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID',act_fn=act_fn)(x)
            x = x.squeeze(axis=2)
            x = x.squeeze(axis=-1)
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x


    # Generate random input data
    #input_data = jnp.ones(input_shape)
    input_data = jax.random.uniform(jax.random.key(0),shape=input_shape)
    cnn_model = CNN(train_quant=True, use_quantize=use_quantize)
    #TODO - disable the eval model 
    eval_model = CNN(train_quant=False, use_quantize=use_quantize)

    rng = jax.random.PRNGKey(0)
    params = cnn_model.init(rng, input_data)
    test_data = jax.random.uniform(rng,shape=input_shape)
    run_model_vs_tflite(eval_model, input_data, act_bits, use_quantize, params=params)
    run_model_vs_tflite(cnn_model, input_data, act_bits, use_quantize, params=params)
    save_and_load_model(cnn_model, params, test_data)

