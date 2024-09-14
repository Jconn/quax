import jax
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
from quax.quax_config import OpConfig
import numpy as np
from jax.core import ShapedArray
from flax.linen.dtypes import promote_dtype
from aqt.jax.v2.aqt_tensor import QTensor

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

class Operation(Enum):
    UNKNOWN = 0 
    FC = 1
    QUANTIZE = 2
    CONV = 3 
    TANH = 4
    ACTIVATION = 5

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
    return marker_p.bind(unwrapped(x), graph_id = graph_id, op_id=op_id)

def marker_impl(x, graph_id, op_id):
    return x

def marker_abstract_eval(xs, graph_id, op_id):
    return core.ShapedArray(xs.shape, xs.dtype)

marker_p.def_impl(marker_impl)
marker_p.def_abstract_eval(marker_abstract_eval)


class Quantize(nn.Module):
    bits: int
    quantized: bool
    op_type: Operation = Operation.QUANTIZE

    @nn.compact
    def __call__(self, x):
        marker_id = id_gen.next()
        marker_prim(x, marker_id, self.op_type.value)
        cfg = OpConfig(self.bits,self.bits, enabled = self.quantized)

        self.sow('quax', 'start',marker_id)
        self.sow('quax', 'bits', {'out': self.bits})
        #calibration axes specify which axes you should calibrate over
        #TODO seems like any activation quantization should be across every axis 
        #but not positive
        x = cfg.quantize(self, x, calibration_axes = [x for x in range(1, x.ndim)])
        marker_prim(x, marker_id, self.op_type.value)
        return x 


def is_quantized(x):
    return isinstance(x, QTensor)

def unwrapped(x): 
    if is_quantized(x):
        return x.dequant()
    else:
        return x

def qtensor_reshape(shape, x):
    qval = x.qvalue.reshape(shape)
    assert np.prod(x.scale[0].shape) == 1, "need single scale for reshape"
    new_scales = [ scale.reshape((1,) * qval.ndim) for scale in x.scale]
    assert x.scale_t is None, "don't know how to handle this"
    #TODO - scale_t reshape
    new_qtensor = QTensor(qvalue= qval, scale=new_scales,scale_t = None, dequant_dtype = x.dequant_dtype)
    return new_qtensor

def reshape(shape, x):
    if is_quantized(x):
        return qtensor_reshape(shape, x) 
    return x.reshape(shape)

def sigmoid(x, out_bits):
    quantized = is_quantized(x)
    x = Activation(out_bits, nn.sigmoid, quantized = quantized)(x) 
    return x 

def tanh(x, out_bits):
    quantized = is_quantized(x)
    x = Activation(out_bits, nn.tanh, quantized = quantized)(x) 
    return x 

class Activation(nn.Module):
    lhs_bits: int
    act_fn: nn.activation
    quantized: bool
    op_type: Operation = Operation.ACTIVATION

    @nn.compact
    def __call__(self, x):
        #x = x.dequant()
        marker_id = id_gen.next()
        marker_prim(x, marker_id, self.op_type.value)
        self.sow('quax', 'start',marker_id)
        #keep rhs bits here for fun
        self.sow('quax', 'bits', {'lhs': self.lhs_bits, 'rhs': self.lhs_bits, 'out': self.lhs_bits})
        quantized = is_quantized(x)
        cfg = OpConfig(self.lhs_bits, self.lhs_bits, enabled = quantized)
        if quantized:
            x = x.dequant()
        x = self.act_fn(x)
        self.sow('quax', 'end', marker_id)
        x = cfg.quantize(self, x, calibration_axes = [x for x in range(1, x.ndim)], po2_scaling = True)
        marker_prim(x, marker_id, self.op_type.value)
        return x


class QDense(nn.Module):
    features: int
    lhs_bits: int
    rhs_bits: int
    quantized: bool
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    op_type: Operation = Operation.FC
    act_fn: nn.activation = None 

    @nn.compact
    def __call__(self, x):
        allowed_acts = [nn.relu, nn.relu6, None]
        assert self.act_fn in allowed_acts, f"can't fuse act fn {self.act_fn}"
        #TODO - assignment means here
        cfg = OpConfig(self.lhs_bits,self.rhs_bits, enabled = self.quantized)
        marker_id = id_gen.next()
        marker_prim(x, marker_id, self.op_type.value)

        #x = x.dequant()
        self.sow('quax', 'start',marker_id)
        self.sow('quax', 'bits', {'lhs': self.lhs_bits, 'rhs': self.rhs_bits, 'out': self.lhs_bits})

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
          bias = cfg.bias_quantize(self, bias, calibration_axes = -1)
        else:
          bias = None

        dot_general = cfg.dot_general()
        x = dot_general(
          x,
          kernel,
          (((x.ndim - 1,), (0,)), ((), ())),
          precision=None,
        )
        if bias is not None:
            #bias has been quantized and dequantized by this point 
            x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))
        if self.act_fn:
            x = self.act_fn(x)


        qx = cfg.quantize(self, x, calibration_axes=-1)

        marker_prim(qx, marker_id, self.op_type.value)
        self.sow('quax', 'end', marker_id)
        return qx 

class QConv(nn.Module):
  """Convolution Module wrapping ``lax.conv_general_dilated``.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. An integer will be
      interpreted as a tuple of the single integer.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), or a sequence of ``n`` ``(low,
      high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same padding
      in all dims and assign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs``
      (default: 1). Convolution with input dilation ``d`` is equivalent to
      transposed convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """

  features: int
  kernel_size: int | Sequence[int]
  lhs_bits: int
  rhs_bits: int
  quantized: bool
  strides: None | int | Sequence[int] = 1
  padding: PaddingLike = 'SAME'
  input_dilation: None | int | Sequence[int] = 1
  kernel_dilation: None | int | Sequence[int] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Array | None = None
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  op_type: Operation = Operation.CONV

  @property
  def shared_weights(self) -> bool:  # type: ignore
    """Defines whether weights are shared or not between different pixels.

    Returns:
      ``True`` to use shared weights in convolution (regular convolution).
      ``False`` to use different weights at different pixels, a.k.a.
      "locally connected layer", "unshared convolution", or "local convolution".

    """
    return True

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features)``.
        This is the channels-last convention, i.e. NHWC for a 2d convolution and
        NDHWC for a 3D convolution. Note: this is different from the input convention
        used by ``lax.conv_general_dilated``, which puts the spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """
    cfg = OpConfig(self.lhs_bits,self.rhs_bits, enabled = self.quantized)
    marker_id = id_gen.next()
    marker_prim(x, marker_id, self.op_type.value)

    #x = x.dequant()
    self.sow('quax', 'start',marker_id)
    self.sow('quax', 'bits', {'lhs': self.lhs_bits, 'rhs': self.rhs_bits, 'out': self.lhs_bits})

    kernel_size: Sequence[int]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(
      x: int | Sequence[int] | None,
    ) -> tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    #TODO - batch dimension reshaping
    #num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    #if num_batch_dimensions != 1:
    #  input_batch_shape = inputs.shape[:num_batch_dimensions]
    #  total_batch_size = int(np.prod(input_batch_shape))
    #  flat_input_shape = (total_batch_size,) + inputs.shape[
    #    num_batch_dimensions:
    #  ]
    #  inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)

    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = flax.linen.linear.canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: list[tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      #TODO - quantized / not handling
      x = jnp.pad(x, pads, mode='wrap')

      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      #TODO - quantized / not handling
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    #TODO - private usage
    dimension_numbers = flax.linen.linear._conv_dimension_numbers(x.shape)
    in_features = jnp.shape(x)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
        in_features // self.feature_group_count,
        self.features,
      )

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
          '`lax.conv_general_dilated_local` does not support '
          f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      #TODO - deciding to use unquantized version here
      #conv_general_dilated = cfg.conv_general_dilated()
      conv_output_shape = eval_shape(
        lambda lhs, rhs: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
          lhs=lhs,
          rhs=rhs,
          window_strides=strides,
          padding=padding_lax,
          dimension_numbers=dimension_numbers,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
        ),
        unwrapped(x),
        ShapedArray(kernel_size + (in_features, self.features), x.dtype),
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (
        np.prod(kernel_size) * in_features,
        self.features,
      )

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.param(
      'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
      bias = cfg.bias_quantize(self, bias, calibration_axes = -1)
    else:
      bias = None


    #inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      conv_general_dilated = cfg.conv_general_dilated()
      y = conv_general_dilated(
        x,
        kernel,
        strides,
        padding_lax,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision,
      )
    else:
        assert False, "case not implemented" 
        y = lax.conv_general_dilated_local(
          lhs=x,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision,
        )

    if self.use_bias:
        bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
        y += bias

    qy = cfg.quantize(self, y, calibration_axes=[x for x in range(1, y.ndim)])
    marker_prim(qy, marker_id, self.op_type.value)
    self.sow('quax', 'end', marker_id)

    #TODO handle this case
    #if num_batch_dimensions != 1:
    #  output_shape = input_batch_shape + y.shape[1:]
    #  y = jnp.reshape(y, output_shape)

    return qy
