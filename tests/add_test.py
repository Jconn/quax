import pytest
import io
import jax
import jax.numpy as jnp
import tensorflow as tf
from quax.quax import Quantize, QDense, Dequantize, QModule, sigmoid
from flax import linen as nn
import numpy as np
from quax.quax_utils import bits_to_type
from quax.jax2tflite import FBB
from base import run_model_vs_tflite, save_and_load_model

@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("weight_bits", [8])
@pytest.mark.parametrize("input_shape", [(1,7), (2,15), (10,35)])
@pytest.mark.parametrize("use_quantize", [False, True])

def test_add(act_bits, weight_bits, input_shape,use_quantize):
    # Create a small CNN model
    class ScalarAdd(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            x = 1.0 + x
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x

    class VectorAdd(QModule):
        use_quantize: bool
        @nn.compact
        def __call__(self, x):
            x = Quantize(bits=act_bits, to_tflite=self.use_quantize)(x)
            x1 = QDense(features=input_shape[-1],lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=False,act_fn=None)(x)
            x2 = QDense(features=input_shape[-1],lhs_bits=act_bits, rhs_bits=weight_bits, use_bias=False,act_fn=None)(x)
            x = x1 + x2
            x = Dequantize(to_tflite=self.use_quantize)(x)
            return x
    def test_model(model):
        fc_model = model(train_quant=True, use_quantize=use_quantize)

        # Generate random input data
        rng = jax.random.PRNGKey(0)
        input_data = jax.random.uniform(rng,shape=input_shape)

        params = fc_model.init(rng, input_data)
        test_data = jax.random.uniform(rng,shape=input_shape)

        run_model_vs_tflite(fc_model, input_data, act_bits, use_quantize, tolerance = 1.01 if act_bits==16 else 0)
        save_and_load_model(fc_model, params, test_data)
    test_model(ScalarAdd)
    test_model(VectorAdd)
