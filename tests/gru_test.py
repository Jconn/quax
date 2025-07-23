import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, GRUCell, Dequantize, QModule
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB
from base import run_model_vs_tflite

@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("features", [31, 121])
@pytest.mark.parametrize("input_shape", [(1, 100), (1,7), (2,2)])
@pytest.mark.parametrize("use_quantize", [False, True])


def test_gru(act_bits, weight_bits, features, input_shape,use_quantize):
    # Create a small CNN model
    class FC(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            x, x_rec = GRUCell(lhs_bits=act_bits, rhs_bits=weight_bits)(x, x)
            x,x_rec = GRUCell(lhs_bits=act_bits, rhs_bits=weight_bits)(x,x_rec)
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x

    fc_model = FC(train_quant=True, use_quantize=use_quantize)

    # Generate random input data
    #input_data = jnp.ones(input_shape)
    input_data = jax.random.uniform(jax.random.key(0),shape=input_shape)

    run_model_vs_tflite(fc_model, input_data, act_bits, use_quantize)

