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
    use_zp: bool = None
    @abc.abstractmethod
    def calibrate(self, qx, x, stored_calibration):
        pass

@aqt_utils.flax_slots_kw_only_dataclass
class PassthroughCalibrator(Calibrator):
    scale: int
    zero_point: int
    def calibrate(self, qx, x, stored_calibration):
        #TODO - array shaping should be normalized here
        self.scale = jnp.array(self.scale, dtype=x.dtype)
        self.zero_point = jnp.array(self.zero_point, dtype=x.dtype)
        return self.scale, self.zero_point

@aqt_utils.flax_slots_kw_only_dataclass
class AbsMaxCalibrator(Calibrator):
    def calibrate(self, qx, x, stored_calibration):
        if self.use_zp is None:
            default_calibrator = calibrator_from_bits(qx.bits)
            return default_calibrator(qx, x)
        else:
            return min_max_calibrator(qx, x, use_zp = self.use_zp)

@aqt_utils.flax_slots_kw_only_dataclass
class MovingAverageAbsMaxCalibrator(Calibrator):
    decay: float = 0.05
    def calibrate(self, qx, x, stored_calibration):
        stored_scale, stored_zp = stored_calibration

        #TODO - zero point mess
        if self.use_zp is None:
            if qx.bits <= 8:
                self.use_zp = True
            default_calibrator = calibrator_from_bits(qx.bits)
            current_cal = default_calibrator(qx, x)
        else:
            current_cal = min_max_calibrator(qx, x, use_zp = self.use_zp)

        if stored_scale == None or stored_zp == None:
            # If no stored calibration, just return the current one
            return current_cal

        current_scale, current_zp = current_cal
        scale = (1.0 - self.decay) * stored_scale + (self.decay) * current_scale
        if self.use_zp == True:
            zero_point = self.decay * stored_zp + (1 - self.decay) * current_zp
        else:
            zero_point = 0.0
        return scale, zero_point



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
    max_val = jnp.max(x, axis=qx.calibration_axes, keepdims=True)
    min_val = jnp.min(x, axis=qx.calibration_axes, keepdims=True)
    mid_point = (max_val + min_val)/2
    if use_zp:
        x = x - mid_point
    abs_max = jnp.max(jnp.abs(x), axis=qx.calibration_axes, keepdims=True)

    bound = abs_max
    bound = jnp.where(bound == 0.0, jnp.ones_like(bound), bound)
    scale = bound / qx.qx_numerics.get_quant_bound()

    scale = ceil_to_po2(scale) if qx.po2_scaling else scale
    #TODO - zp 
    if use_zp:
        zp = mid_point / scale
        zp = zp.astype(jnp.int8)
    else:
        zp = jnp.zeros(scale.shape)
    scale = jnp.squeeze(scale)
    zp = jnp.squeeze(zp)
    return scale, -zp 


def calibrator_from_bits(bits):
    if bits <= 8:
        return default_8bit_calibrator 
    elif bits <= 16:
        return default_16bit_calibrator

