import jax
import jax.numpy as jnp
from jax import core
import flax.linen as nn
from aqt.jax.v2.aqt_quantizer import Quantizer 
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2 import calibration
from aqt.jax.v2.aqt_dot_general import MultiTensor
from flax.linen import initializers
from aqt.jax.v2 import config as aqt_config
from jax import lax
from enum import Enum
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)

class Operation(Enum):
    UNKNOWN = 0 
    FC = 1

class MID:
    def __init__(self):
        self.current_id = 0
    def reset(self):
        self.current_id = 0
    def next(self):
        next_id = self.current_id
        self.current_id += 1
        return next_id
id_gen = MID()

default_kernel_init = initializers.lecun_normal()
marker_p = core.Primitive("marker")

def marker_prim(x, graph_id, op_id):
    return marker_p.bind(x, graph_id = graph_id, op_id=op_id)

def marker_impl(x, graph_id, op_id):
    return x

def marker_abstract_eval(xs, graph_id, op_id):
    return core.ShapedArray(xs.shape, xs.dtype)

marker_p.def_impl(marker_impl)
marker_p.def_abstract_eval(marker_abstract_eval)

bq = Quantizer(
    numerics=int_numerics.IntNumerics(
        bits=16,
        preserve_zero=True,
        preserve_max_val=True,
        clip=True,
        clip_gradient=True,
        round=True,
        noise_fn=None,
        dtype = jnp.int16,
    ),
    calib_shared_axes=-1,
    scale_stop_grad=True,
    calibration=calibration.AbsMaxCalibration,
    po2_scale=False,
    context=aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=0))
bq.init_calibration()

q = Quantizer(
    numerics=int_numerics.IntNumerics(
        bits=8,
        preserve_zero=True,
        preserve_max_val=True,
        clip=True,
        clip_gradient=True,
        round=True,
        noise_fn=None,
        dtype = jnp.int8,
    ),
    calib_shared_axes=-1,
    scale_stop_grad=True,
    calibration=calibration.AbsMaxCalibration,
    po2_scale=False,
    context=aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=0))

q.init_calibration()

class Quantize(nn.Module):
    aqt_cfg: aqt_config.DotGeneral
    calibration_axes: int
    @nn.compact
    def __call__(self, x):
        if self.aqt_cfg == None:
            return x
        qx,_ =  q.quant(x, calibration_axes = self.calibration_axes)

        def initializer():
          return qx.without_qvalue() 
        self.variable('aqt', 'frozen', initializer)
        return qx

class QDense(nn.Module):
    features: int
    aqt_cfg: aqt_config.DotGeneral
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    op_type: Operation = Operation.FC

    @nn.compact
    def __call__(self, x):
        #TODO - assignment means here
        id_gen.reset()
        marker_id = id_gen.next()
        #x = x.dequant()
        self.sow('quax', 'start',marker_id)
        self.sow('quax', 'bits', {'lhs': 8, 'rhs': 8, 'out': 8})
        if self.aqt_cfg:
            tmp = marker_prim(x.dequant(), marker_id, self.op_type.value)
        else:
            tmp = marker_prim(x, marker_id, self.op_type.value)
        #x = x.dequant()

        kernel = self.param(
          'kernel',
          self.kernel_init,
          (jnp.shape(x)[-1], self.features),
          self.param_dtype,
        )
        if self.use_bias:

          bias = self.param(
            'bias', self.bias_init, (self.features,), self.param_dtype
          )
          if self.aqt_cfg:
            bias_quant,_ =  bq.quant(bias, calibration_axes = -1)
            def initializer():
              return bias_quant
            self.variable('aqt', 'bias', initializer)
            bias = bias_quant.dequant()  
        else:
          bias = None


        if self.aqt_cfg is not None:
          dot_general = self.aqt_cfg()
        else:
          dot_general = lax.dot_general
        #import pdb; pdb.set_trace()
        x = dot_general(
          x,
          kernel,
          (((x.ndim - 1,), (0,)), ((), ())),
          precision=None,
        )
        if bias is not None:
          #how do I handle the bias man..
          #I have the dequantized aqt tensor
          #from the dg
          #what does this mean
          #I quantize and then dequantize the bias tensor and sum the two?
          x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))

        if self.aqt_cfg is not None:
            qx,_ =  q.quant(x, calibration_axes = -1)

            def initializer():
              return qx.without_qvalue() 
            self.variable('aqt', 'output', initializer)
        else:
            qx = x

        tmp = marker_prim(x, marker_id, self.op_type.value)
        self.sow('quax', 'end', marker_id)
        return qx 




