import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jax import core
import flax.linen as nn
import flax
from aqt.jax.v2.aqt_quantizer import Quantizer 
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2 import calibration
from aqt.jax.v2.aqt_dot_general import MultiTensor
from flax.linen import initializers
from aqt.jax.v2 import config as aqt_config
from jax import eval_shape, lax
from enum import Enum
from quax.quax_config import OpConfig, quantizer, requantizer
from quax.quaxpr import quaxpr_prim, Operation,ActivationType, pytree_fc, quaxpr_default, quaxpr_multiarg, quaxpr_functional, quaxpr_unquant_prim, AppendedActivation
import numpy as np
from jax.core import ShapedArray
from flax.linen.dtypes import promote_dtype
from quax.quax_utils import bits_to_type
from aqt.jax.v2.aqt_tensor import QTensor
from jax.interpreters import ad
from aqt.jax.v2 import aqt_tensor
from functools import partial
from flax.typing import (
  Any,
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
  Sequence
)

from aqt.jax.v2.numerics import numerics
from aqt.jax.v2.numerics import int_numerics
import abc

@aqt_utils.flax_slots_kw_only_dataclass
class Calibrator(abc.ABC):
    @abc.abstractmethod
    def calibrate(self, qx, x):
        pass

@aqt_utils.flax_slots_kw_only_dataclass
class PassthroughCalibrator(Calibrator):
    def calibrate(self, qx, x):
        return passthrough(qx, x)

@aqt_utils.flax_slots_kw_only_dataclass
class AbsMaxCalibrator(Calibrator):
    use_zp: bool = None
    def calibrate(self, qx, x):
        if self.use_zp is None:
            default_calibrator = calibrator_from_bits(qx.bits)
            return default_calibrator(qx, x)
        else:
            return min_max_calibrator(qx, x, use_zp = self.use_zp)



def ceil_to_po2(scale: jnp.ndarray) -> jnp.ndarray:
  # With floor the biggest value (we are using jnp.max) is in the range of
  # clipping and therefore have a correct gradient.
  scale = 2 ** jnp.floor(jnp.log2(jax.lax.reciprocal(scale)))
  scale = jax.lax.reciprocal(scale)
  return scale


def default_8bit_calibrator(qx, x):
    return min_max_calibrator(qx, x, use_zp = True)

def default_16bit_calibrator(qx, x):
    return min_max_calibrator(qx, x, use_zp = False)

def min_max_calibrator(qx, x, use_zp = False):
    #if use_zp:
    #    zp = jnp.mean(x, axis=qx.calibration_axes, keepdims=True)
    #    zp = jnp.zeros([1], dtype = x.dtype)
    #else:
    #    zp = jnp.zeros([1], dtype = x.dtype)
    abs_max = jnp.max(jnp.abs(x), axis=qx.calibration_axes, keepdims=True)
    bound = abs_max
    bound = jnp.where(bound == 0.0, jnp.ones_like(bound), bound)
    scale = bound / qx.qx_numerics.get_quant_bound()

    scale = ceil_to_po2(scale) if qx.po2_scaling else scale
    #TODO - zp 
    zp = jnp.zeros(scale.shape)
    #TODO - fix zero point usage
    #zp = jnp.array([zp], dtype=scale.dtype)
    return scale, zp 

def passthrough(qx, x):
    return qx.scale, qx.zp 

def calibrator_from_bits(bits):
    if bits <= 8:
        return default_8bit_calibrator 
    elif bits <= 16:
        return default_16bit_calibrator

