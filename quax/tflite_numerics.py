"""Numerics for TFLite integer types"""

from typing import Any
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
from jax import lax
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class IntAsymmetric(numerics.AqtNumerics):
  """ASymmetric numerics for sint8, sint4, etc that follows tflite numerics"""
  bits: int
  clip: bool
  clip_gradient: bool
  dtype: None | Any = None

  # pylint: disable=line-too-long
  # Verifying the correctness of these functions amounts to verifying this table:
  # if preserve_zero == F, zero might be rounded either to [-1, 0] bucket or to [0, 1] bucket
  # preserve_zero, preserve_max_val, 8b, 2b, 1b
  # F, F, 128.0, 2.0, 1.0  # bucket count is even; map onto the far edge of the last bucket
  # F, T, 127.5, 1.5, 0.5  # bucket count is even; map onto the center of the last bucket
  # T, F, 127.5, 1.5, 0.5  # bucket count is odd;  map onto the far edge of the last bucket
  # T, T, 127.0, 1.0, 0.0  # bucket count is odd;  map onto the center of the last bucket
  # pylint: enable=line-too-long

  def get_edge_of_last_int_bucket(self):
    ret = 2.0 ** (self.bits - 1)
    if self.preserve_zero:
      # Lose one bucket.
      ret -= 0.5
    return ret


  def get_quant_bound(self):
      return self._get_fwd_clip_bound() - self._get_bwd_clip_bound()

  def _get_fwd_clip_bound(self):
    # If we are not rounding, we just clip to bucket edges.
      return (2.0 ** (self.bits - 1)) - 1

  def _get_bwd_clip_bound(self):
      return -2.0 ** (self.bits - 1) 

  def get_dtype(self):
    return self.dtype

  def vjp_fwd(self, x, context):
    """Forward pass."""
    res = (x,)
    input_dtype = x.dtype
    assert self.bits <= 22, 'Too many bits, float32 has less precision.'

    fwd_clip_bound = self._get_fwd_clip_bound()
    bwd_clip_bound = self._get_bwd_clip_bound()
    if self.clip:
        x = jnp.clip(x, bwd_clip_bound, fwd_clip_bound)
    x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    # Maybe cast: return dtype is either int or the input dtype
    dtype = self.get_dtype()
    x = x.astype(dtype if dtype is not None else input_dtype)
    return x, res

  def vjp_bwd(self, res, grad):
    # Gradient of the clip function.
    # For boundary values we will have full gradient.
    # When using abs(max(x)) scaling, x is always in the interior, and the
    # gradient clip is always 1. So, we can always set clip_gradient to false.
    # However, other types of scaling may result in x being outside (i.e., there
    # is clipping). In that case it may be desirable to make the gradient zero.
    ret = grad
    if self.clip_gradient:
      (x,) = res
      fwd_clip_bound = self._get_fwd_clip_bound()
      bwd_clip_bound = self._get_bwd_clip_bound()
      ret *= (bwd_clip_bound <= x) * (x <= fwd_clip_bound)
    return (ret, None)
