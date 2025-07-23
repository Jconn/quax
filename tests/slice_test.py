import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, QDense, Dequantize, QModule
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB
from base import run_model_vs_tflite, save_and_load_model

@pytest.mark.parametrize("act_bits", [8])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [121])
@pytest.mark.parametrize("input_shape", [(1, 100), ])
@pytest.mark.parametrize("use_quantize", [False, True])
@pytest.mark.parametrize("use_bias", [False])
@pytest.mark.parametrize("use_relu", [False])

def test_slice(act_bits, weight_bits, features, input_shape,use_quantize, use_bias, use_relu):
    # Create a small CNN model
    class FCSlice(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            act_fn = nn.relu if use_relu else None 
            x = x[:, 4:50]
            x = QDense(features=features,lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=use_bias,act_fn=act_fn)(x)
            x = x[:, :10]
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x

    fc_model = FCSlice(train_quant=True, use_quantize=use_quantize)

    # Generate random input data
    #input_data = jnp.ones(input_shape)
    rng = jax.random.PRNGKey(0)
    input_data = jax.random.uniform(rng,shape=input_shape)

    params = fc_model.init(rng, input_data)
    test_data = jax.random.uniform(rng,shape=input_shape)

    run_model_vs_tflite(fc_model, input_data, act_bits, use_quantize)
    save_and_load_model(fc_model, params, test_data)

