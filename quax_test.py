import jax
import jax.numpy as jnp
from flax import linen as nn
from quax.jax2tflite import FBB
from quax.quax import QDense, Quantize, QConv, id_gen
from quax import quax
import tensorflow as tf
from jax import core
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax import aqt_flax_calibration
from aqt.jax.v2.aqt_dot_general import CalibrationMode
import functools

def config_wrapper(fwd_bits, bwd_bits):
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    aqt_config.set_fwd_calibration_mode(cfg, rhs_calibration_mode=CalibrationMode.REMAINING_AXIS)
    return cfg

if __name__ == "__main__":
    class testModel(nn.Module):
        quantized: bool 
        @nn.compact
        def __call__(self, x):
            id_gen.reset()
            act_bits = 16 
            weight_bits = 8
            x = Quantize(bits=act_bits, quantized = self.quantized)(x)
            x = QConv(features=9, kernel_size = (1,3),lhs_bits = act_bits, rhs_bits = weight_bits, quantized= self.quantized)(x)
            x = quax.tanh(x, act_bits)
            x = quax.reshape([1,-1], x)
            x = QDense(features=10, lhs_bits = act_bits, rhs_bits = weight_bits, quantized= self.quantized, act_fn = nn.relu)(x)
            x = quax.sigmoid(x, act_bits)
            if self.quantized: 
                x = x.dequant()
            return x
    model = testModel(quantized=True)
    server = testModel(quantized=False)
    x = jnp.ones([1,3,3,10])
    params = model.init({'params': jax.random.key(0)},x)

    #params['aqt']['QDense_0']['Dense_0']['AqtDotGeneral_0']
    converter = FBB()
    tflite = converter.convert(server, params, x)
    with open('manual_model.tflite', 'wb') as f:
        f.write(tflite)
    
    interpreter = tf.lite.Interpreter(model_path="manual_model.tflite")
    #interpreter = tf.lite.Interpreter(model_path="/home/johnconn/fc_32_64_f32.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input Details:", input_details)
    print("Output Details:", output_details)

    # Print all tensor details
    tensor_details = interpreter.get_tensor_details()
    for detail in tensor_details:
        print(detail)
