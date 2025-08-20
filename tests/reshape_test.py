import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, QDense,QConv, Dequantize, QModule
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB
from base import run_model_vs_tflite, save_and_load_model

@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [32, 84])
@pytest.mark.parametrize("input_shape", [(1,12), (2,2)])
@pytest.mark.parametrize("use_quantize", [True])
@pytest.mark.parametrize("use_bias", [True])
@pytest.mark.parametrize("use_relu", [True])

def test_reshape(act_bits, weight_bits, features, input_shape,use_quantize, use_bias, use_relu):
    # Create a small CNN model
    class ReshapeModel(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            act_fn = nn.relu if use_relu else None 
            x = QDense(features=features,lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=use_bias,act_fn=act_fn)(x)

            x = x.reshape([x.shape[0], x.shape[1]//4,2, 2])
            x = QConv(features=2, strides=(1,1), kernel_size=(1,1),
                      lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=True, padding='VALID',act_fn=act_fn)(x)
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x

    reshape_model = ReshapeModel(train_quant=True, use_quantize=use_quantize)

    rng = jax.random.PRNGKey(0)
    input_data = jax.random.uniform(rng,shape=input_shape)

    params = reshape_model.init(rng, input_data)
    test_data = jax.random.uniform(rng,shape=input_shape)

    run_model_vs_tflite(reshape_model, input_data, act_bits, use_quantize)
    save_and_load_model(reshape_model, params, test_data)

