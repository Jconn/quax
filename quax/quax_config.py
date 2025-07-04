'''
what is the goal of this specification?
want to specify the number of bits of operation 
output bits is equal to lhs bits

'''
from dataclasses import dataclass
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.aqt_quantizer import Quantizer 
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils as aqt_utils
import jax
import jax.numpy as jnp
from quax.quax_utils import bits_to_type
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import config as aqt_config
import functools
from jax import lax
from typing import Literal, Optional, TypeAlias, Union
from aqt.jax.v2.aqt_dot_general import LocalAqt
from aqt.jax.v2.aqt_dot_general import dot_general_make
from aqt.jax.v2.aqt_dot_general import DotGeneral
from aqt.jax.v2.aqt_tensor import QTensor
from aqt.jax.v2.aqt_dot_general import MultiTensor 
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.calibration import Calibration 

@aqt_utils.flax_slots_kw_only_dataclass
class InheritedCalbration(Calibration):
  """Calibration with a constant per-tensor or per-channel value."""

  scale: jnp.ndarray | float
  bias: None | jnp.ndarray | float = None

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes,
      numerics_,
      context,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    dtype = self.dtype if self.dtype is not None else x.dtype
    if self.bias is None:
      bias = []
    elif jnp.isscalar(self.bias) or isinstance(self.bias, float):
      # floats are scalars, but pytype can't infer that.
      bias = [jnp.full(x.shape, self.bias, x.dtype)]
    else:
      bias = [self.bias.astype(dtype)]
    #sparsity now i guess..
    return [self.scale.astype(dtype)], bias, None



def requantizer(bits, scale, po2_scaling = False):
    #TODO - how to deal with context
    dtype = bits_to_type(bits)
    scale = scale[0]
    #TODO bias
    quant_calib = functools.partial(
        InheritedCalbration,
        po2_scale=po2_scaling,
        scale =scale 
    )
    quant = Quantizer(
        numerics=int_numerics.IntSymmetric(
            bits=bits,
            preserve_zero=True,
            preserve_max_val=False,
            clip=True,
            clip_gradient=False,
            round=True,
            noise_fn=None,
            dtype = dtype,
        ),
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=quant_calib,
        context=aqt_utils.Context(key=None, train_step=None)
        )
    quant.init_calibration()
    return quant

def quantizer(bits, po2_scaling = False, scale=None):
    #TODO - how to deal with context
    dtype = bits_to_type(bits)
    if scale:
        if not isinstance(scale, jnp.ndarray):
            scale = jnp.array(scale)

        quant_calib = functools.partial(
            InheritedCalbration,
            po2_scale=po2_scaling,
            scale =scale 
        )

    else:
        quant_calib = functools.partial(
            calibration.AbsMaxCalibration,
            po2_scale=po2_scaling,
        )
    quant = Quantizer(
        numerics=int_numerics.IntSymmetric(
            bits=bits,
            preserve_zero=True,
            preserve_max_val=False,
            clip=True,
            clip_gradient=False,
            round=True,
            noise_fn=None,
            dtype = dtype,
        ),
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=quant_calib,
        context=aqt_utils.Context(key=None, train_step=None)
        )
    quant.init_calibration()
    return quant

def quantized_dg_cfg(
    *,
    lhs_bits: Optional[int] = 8,
    rhs_bits: Optional[int] = 8,
    bwd_bits: Optional[int] = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: Optional[bool] = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: Optional[bool] = None,
    vjp_rhs_stochastic_rounding: Optional[bool] = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  cfg = dot_general_make(
      lhs_bits=lhs_bits,
      rhs_bits=rhs_bits,
      bwd_bits=bwd_bits,
      use_fwd_quant=use_fwd_quant,
      dlhs_local_aqt=dlhs_local_aqt,
      drhs_local_aqt=drhs_local_aqt,
  )

  # Stochastic Rounding
  # These 3 variables are used to ensure we don't mix
  # old and new style of SR configuration.
  old_style_sr_config = use_stochastic_rounding is not None
  new_style_sr_config_lhs = vjp_lhs_stochastic_rounding is not None
  new_style_sr_config_rhs = vjp_rhs_stochastic_rounding is not None
  assert new_style_sr_config_lhs == new_style_sr_config_rhs, (
      'if you use new style SR config (vjp_xhs_stochastic_rounding), do pass'
      ' both lhs and rhs explicitely.'
  )
  assert new_style_sr_config_lhs != old_style_sr_config

  true = True  # A crude way to get around g-explicit-bool-comparison warning

  assert not (vjp_lhs_stochastic_rounding and vjp_rhs_stochastic_rounding), (
      'This config is buggy when you set both to True. Contact lew@ or use'
      ' config_v3'
  )

  # By default use jax.uniform for stochastic rounding
  if use_stochastic_rounding == true:
    aqt_config.set_stochastic_rounding(cfg, True, True, 'jax.uniform')

  if vjp_lhs_stochastic_rounding == true:
    aqt_config.set_stochastic_rounding(cfg, True, False, 'jax.uniform')

  if vjp_rhs_stochastic_rounding == true:
    aqt_config.set_stochastic_rounding(cfg, False, True, 'jax.uniform')

  if use_dummy_static_bound:
    aqt_config.set_static_bound(cfg, 1.0)
  
  #force dummy gradient allow since we don't update lhs quant details 
  cfg.fwd.allow_dummy_gradient_into_qtensor = True
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'

  return cfg
@dataclass
class OpConfig:
    lhs_bits: int
    rhs_bits: int
    enabled: bool
    calib_shared_axes: int = -1

    def conv_general_dilated(self): 
        if self.enabled:
            #reusing dg config for conv for now..
            aqt_bit_cfg =  quantized_dg_cfg(lhs_bits=self.lhs_bits,rhs_bits=self.rhs_bits, bwd_bits=self.lhs_bits)
            aqt_conv_dilated = functools.partial(
                aqt_flax.AqtConvGeneralDilated,
                aqt_bit_cfg,
              lhs_quant_mode=aqt_utils.QuantMode.TRAIN,
              rhs_quant_mode=aqt_utils.QuantMode.TRAIN,
                tiling_cfg=None,
                lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
                rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            )
            dilated_conv = aqt_conv_dilated()
        else:
            dilated_conv = lax.conv_general_dilated
        return dilated_conv
    def bias_quantizer(self):
        #TODO - how to deal with context
        #TODO the calibration axes should be parameterized I thin
        lhs_type = bits_to_type(self.lhs_bits)
        quant = Quantizer(
            numerics=int_numerics.IntNumerics(
                bits=16,
                preserve_zero=True,
                preserve_max_val=True,
                clip=True,
                clip_gradient=True,
                round=True,
                noise_fn=None,
                dtype = lhs_type,
            ),
            calib_shared_axes=self.calib_shared_axes,
            scale_stop_grad=True,
            calibration=calibration.AbsMaxCalibration,
            po2_scale=False,
            context=aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=0))
        quant.init_calibration()
        return quant

    def quantizer(self, po2_scaling = False, bits = None):
        #TODO - how to deal with context
        #TODO the calibration axes should be parameterized I thin
        if bits is None:
            bits = self.lhs_bits
        lhs_type = bits_to_type(self.lhs_bits)
        quant = Quantizer(
            numerics=int_numerics.IntNumerics(
                bits=bits,
                preserve_zero=True,
                preserve_max_val=False,
                clip=True,
                clip_gradient=False,
                round=True,
                noise_fn=None,
                dtype = lhs_type,
            ),
            calib_shared_axes=None,
            scale_stop_grad=True,
            calibration=calibration.AbsMaxCalibration,
            po2_scale=po2_scaling,
            context=aqt_utils.Context(key=None, train_step=None)
            )
        quant.init_calibration()
        return quant

    def bias_quantize(self, mdl, bias, calibration_axes):
        if not self.enabled:
            return bias 
        bq = self.bias_quantizer()
        bias_quant,_ = bq.quant(bias, calibration_axes = calibration_axes)
        def initializer():
          return bias_quant
        mdl.variable('aqt', 'bias', initializer)
        bias = bias_quant.dequant()

        return bias

