import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jax import core
import flax.linen as nn
import flax
from typing import Callable
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2 import calibration
from flax.linen import initializers
import aqt.jax.v2.numerics as numerics 
from dataclasses import field
from jax import eval_shape, lax
from enum import Enum
from quax.quaxpr import quaxpr_prim, Operation,ActivationType, pytree_fc, quaxpr_default, quaxpr_multiarg, quaxpr_functional, quaxpr_unquant_prim, AppendedActivation
import quax.quantizer as quantizer 
import numpy as np
from jax.core import ShapedArray
from flax.linen.dtypes import promote_dtype
from quax.quax_utils import bits_to_type
from aqt.jax.v2.aqt_tensor import QTensor
from jax.interpreters import ad
from aqt.jax.v2 import aqt_tensor
from quax.quantizer import Calibrator, PassthroughCalibrator, AbsMaxCalibrator
from functools import partial
import logging
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

import copy
model_enroll = None

def quaxtensor_from_scalar(scalar , qx):
    array = jnp.array(scalar)
    new_qx = quantize(array, calibration_axes=[x for x in range(0, scalar.ndim)],
                  bits = qx.bits,
                  calibrator = quantizer.PassthroughCalibrator(),
                  po2_scaling = qx.po2_scale, scale = qx.scale, zero_point = qx.zp)
    return new_qx


def zeros(shape, bits):
    x = jnp.zeros(shape)
    qx = aqt_tensor.zeros(shape, container_dtype = jnp.float32)
    scale = jnp.array([1./(2**bits-1)])
    scale = scale.reshape((1,) + (1,) * (len(shape)-1)) 
    qx.scale.append(scale)
    quaxt = QuaxTensor(x = x, qx = qx, bits = bits)
    return quaxt

def enroll_model(mdl):
    global model_enroll
    model_enroll = mdl

def enrolled_model():
    return model_enroll

def store_activation(mdl, name, qxtensor):
    store_quantized(mdl, name, qxtensor,scale_only=True)

def store_quantized(mdl, name, qxtensor,scale_only=False):
    init_fn = lambda: 0
    reduce_fn = lambda a, b:  b
    #will return true if the store succeeds
    #will fail by default when no longer mutable
    qxcopy = qxtensor
    if scale_only:
        qxcopy = copy.deepcopy(qxcopy)
        qxcopy.x = None 
    return mdl.sow('quax', name, qxcopy,
                  init_fn=init_fn, reduce_fn=reduce_fn)


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
# Define the JVP rule for differentiation
def marker_jvp(primals, tangents, **params):
    x = primals
    t_x = tangents
    y = marker_p.bind(x, **params)
    return y, t_x

ad.defjvp(marker_p, marker_jvp)

class Dequantize(nn.Module):
    '''
    this takes as input a quaxtensor
    returns a numpy tensor
    '''
    op_type: Operation = Operation.DEQUANTIZE

    @nn.compact
    def __call__(self, qx):
        quax_pytree = {}
        quax_pytree['op'] = self.op_type
        #this is unquantized, so we can't use the default quaxpr gen here
        store_activation(self, 'input', qx)
        quaxpr_default(qx, self.op_type,self )
        #TODO - this approach of jaxpr wrapping is flawed
        x = jnp.add(qx.x, 0)
        quaxpr_unquant_prim(x, quax_pytree)
        return x 

class Quantize(nn.Module):
    bits: int
    op_type: Operation = Operation.QUANTIZE
    to_tflite: bool = True 

    @nn.compact
    def __call__(self, x):
        quax_pytree = {}
        quax_pytree['op'] = self.op_type
        quax_pytree['to_tflite'] = self.to_tflite

        quaxpr_unquant_prim(x, quax_pytree)

        qx = quantize(x, calibration_axes=[x for x in range(0, x.ndim)],
                      bits = self.bits,
                      calibrator = quantizer.AbsMaxCalibrator(),
                      po2_scaling = False)


        store_activation(self, 'output', qx)
        quaxpr_default(qx, self.op_type,self )
        return qx 

def requantize_op(x, scale):
    @jax.custom_vjp
    def requant(x,scale):
        unquantized = x.x 
        dtype = bits_to_type(x.bits)
        x.qx.qvalue = (unquantized / scale[0][0]).astype(dtype)
        x.qx.scale = scale
        x.x = x.qx.dequant()
        return x, x.x 

    def requant_fwd(x,scale):
        unquantized = x.x 
        dtype = bits_to_type(x.bits)
        x.qx.qvalue = (unquantized / scale[0][0]).astype(dtype)
        x.qx.scale = scale
        x.x = x.qx.dequant()
        return (x, x.x), 

    def requant_bwd(res, g):
        quant_grad = res
        return quant_grad(g[1])[0],

    requant.defvjp(requant_fwd, requant_bwd)
    qxt, x = requant(x,scale)
    #quaxt = QuaxTensor(x = x, qx = qx, bits = q.numerics.bits)
    return qxt 

def quantize(x, qx=None, **kwargs):
    assert isinstance(x, jnp.ndarray), "x must be jnp array"

    if kwargs.get('calibration_axes') is None:
      #if not specified, assume single quantization for entire tensor
      calibration_axes = [x for x in range(0, x.ndim)]
      kwargs['calibration_axes'] = calibration_axes

    if qx is None:

      if kwargs.get('qx_numerics') is None:
        default_numerics = numerics.int_numerics.IntSymmetric(bits=kwargs.get('bits'),
                                                              preserve_zero=True,
                                                              preserve_max_val=True,
                                                              clip=True,
                                                              clip_gradient=True,
                                                              round=True,
                                                              noise_fn = None)
        kwargs['qx_numerics'] = default_numerics
      qx = QuaxTensor(x=x, **kwargs)
    else:
      qx = inherit_quaxtensor(qx)
    #override calibration axes to make sure prior value isn't actually
    #TODO - remove calibration_axes from tensor, or maybe crush it when a reshape, etc. occurs
    qx.calibration_axes = kwargs.get('calibration_axes')

    #don't trace calibration step since it isn't gradient effected
    qx.calibrate(x)
    #scale, zp = qx.calibrator.calibrate(qx, x)
    #qx = qx.replace(scale=jax.lax.stop_gradient(scale), zero_point=jax.lax.stop_gradient(zp))
    @jax.custom_vjp
    def quant(x):
        x, grad = qx.fake_quant(x)
        return x 

    def quant_fwd(x):
        x, grad = qx.fake_quant(x)
        return x, grad 

    def quant_bwd(res, g):
        quant_grad = res
        return quant_grad(g)[0],

    quant.defvjp(quant_fwd, quant_bwd)
    x = quant(x)
    qx = qx.replace(x=x) 
    return qx

def is_quantized(x):
    return isinstance(x, QTensor)

def unwrapped(x): 
    if is_quantized(x):
        return x.dequant()
    else:
        return x

def inherit_quaxtensor(qx):
    return copy.deepcopy(qx)


#CalibratorFn = Callable[[Any, jnp.ndarray], jnp.ndarray, jnp.ndarray]

@aqt_utils.flax_slots_kw_only_dataclass
class QuaxTensor:
    x: jnp.ndarray
    bits: int
    calibration_axes: Sequence[int]
    qx_numerics: numerics.numerics.AqtNumerics
    calibrator: Calibrator
    scale: jnp.ndarray = None
    zero_point: jnp.ndarray = None
    po2_scaling: bool = False
    def calibrate(self, x):
        scale, zp = self.calibrator.calibrate(self, x)
        self.scale = jax.lax.stop_gradient(scale)
        self.zero_point = jax.lax.stop_gradient(zp)

    def fake_quant(self, orig_x):
        qx = self.quant(orig_x)
        #TODO - do I keep the gradients alive?
        x_q, res = self.qx_numerics.vjp_fwd(qx, context=None)
        quant_grad = jax.tree_util.Partial(self.qx_numerics.vjp_bwd, res)
        x = self.dequant(x_q)
        return x, quant_grad
    
    def quant(self, x):
        #quant(x) = (x + b) /s
        qvalue = x
        qvalue = qvalue + self.zero_point
        s_inv = jax.lax.reciprocal(self.scale)
        s_inv = jnp.where(jnp.isinf(s_inv), jnp.ones_like(s_inv), s_inv)
        qvalue *= s_inv
        return qvalue

    def dequant(self, qvalue):
        #dequant(x) = x*s - b
        x = qvalue * self.scale
        x -= self.zero_point
        return x

    @property 
    def shape(self):
        return self.x.shape

    @property 
    def ndim(self):
        return self.x.ndim

    @property
    def bits(self):
        return self.bits.item()

    def to_np(self):
        np_x = np.array(self.x)
        np_qx = QuaxTensor
        return QuaxTensor(x = np_x, qx = np_qx, bits=self.bits)

    def reshape(self, shape):
        quaxtensor_reshape(shape, self)
        return self

    #TODO - these aren't supposedto mutate object
    def squeeze(self, axis=None):
        quaxtensor_squeeze(self,axis)
        return self

    def transpose(self, axes=None):
        return quaxtensor_transpose(self, axes)

    def unquantized_tensor(self):
        return self.x

    def quantized_tensor(self):
        return self.quant(self.x)

    def __sub__(self, other):
        return self.sub(other, jnp.subtract, Operation.SUB, rsub=False)

    def __rsub__(self, other):
        return self.sub(other, jnp.subtract, Operation.SUB, rsub=True)

    def __add__(self, other):
        return self._apply_op(other, jnp.add, Operation.ADD)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self._apply_op(other, jnp.multiply, Operation.MUL)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, key):
        if isinstance(key, (tuple, slice))  :
            return self.apply_slice(key)
        else:
            raise TypeError("Slicing must use a slice object.")

    def apply_slice(self, key):
        op_type =  Operation.SLICE
        op_name = enrolled_model().scope.default_name("op") 
        mdl = enrolled_model()
        mdl.scope.reserve(op_name)
        op_name = f"element-{op_name}"
        quaxpr_default(self, op_type, mdl, op_name = op_name)
        orig_shape = self.shape

        # Create new sliced arrays instead of modifying in place
        new_x = self.x[key]

        if isinstance(key, slice):
            key = (key)

        #great can only have one ellipsis
        new_key = []
        for idx, kval in enumerate(key):
            if isinstance(kval, type(Ellipsis)):
                num_dims = len(orig_shape)
                ellipse_fill = num_dims - (len(key) - 1) #do not count ellipse in the key size
                for i in range(ellipse_fill):
                    new_key.append(slice(None,None,None))
            elif isinstance(kval, int): 
                #this corrects for any ellipse
                corrected_idx = len(new_key)

                if kval <0:
                    #if -1, it is the last dimension
                    last_idx = orig_shape[corrected_idx] 
                    kval = last_idx - kval
                new_key.append(slice(kval,kval+1,1))
            else:
                new_key.append(kval)
        key = new_key
        new_key = []

        for dim_idx,sl in enumerate(key):
            begin, end, idx = sl.start, sl.stop, sl.step
            if begin is None:
                begin = 0
            if end is None:
                end = orig_shape[dim_idx]
            if idx is None:
                idx = 1
            new_key.append(slice(begin, end, idx))

        
        # Create and return new QuaxTensor with sliced data
        new_qx_tensor = copy.deepcopy(self)
        new_qx_tensor.replace(x = new_x)
        quaxpr_default(new_qx_tensor, op_type, mdl, op_name = op_name, slice_key = new_key)

        return new_qx_tensor

    def sub(self, other, op, op_type, rsub):
        op_name = enrolled_model().scope.default_name("op") 
        enrolled_model().scope.reserve(op_name)
        op_name = f"element-{op_name}"
        has_scalar = False 
        if rsub:
            input1 = other
            input2 = self
        else:
            input1 = self
            input2 = other

        if isinstance(other, QuaxTensor):
            new_x = op(input1.x, input2.x)
            quaxpr_multiarg(input1, input2, op=op_type, mdl=enrolled_model(), op_name = op_name, has_scalar = has_scalar)

        elif jnp.isscalar(other):
            has_scalar = True
            if rsub:
                new_x = op(input1, input2.x)
            else:
                new_x = op(input1.x, input2)

            other = quaxtensor_from_scalar(other, self)
            if rsub:
                input1 = other
            else:
                input2 = other
            scalar_name = f"{op_name}-scalar"
            #TODO - do we know if this is a weight or activation?
            store_quantized(enrolled_model(), scalar_name, other)
            quaxpr_multiarg(input1, input2, op=op_type, mdl=enrolled_model(), op_name = op_name, has_scalar = has_scalar, scalar_idx= 0 if rsub else 1)
        else:
            raise TypeError(f"Unsupported type for operation: {type(other)}")

        #unquantized output
        #default - quantize based on the input tensor?
        qx = quantize(new_x,qx=self)
        #this is the hard part for me
        #need to store 
        store_activation(enrolled_model(), op_name, qx)
        quaxpr_default(qx, op_type, enrolled_model(), op_name = op_name)
        return qx




    def _apply_op(self, other, op, op_type):
        op_name = enrolled_model().scope.default_name("op") 
        enrolled_model().scope.reserve(op_name)
        op_name = f"element-{op_name}"
        #quaxpr_multiarg(self, other, op_type, enrolled_model(), op_name = op_name)
        has_scalar = False 
        if isinstance(other, QuaxTensor):
            new_x = op(self.x, other.x)
            quaxpr_multiarg(self, other, op=op_type, mdl=enrolled_model(), op_name = op_name, has_scalar = has_scalar)
        elif jnp.isscalar(other):
            has_scalar = True
            new_x = op(self.x, other)
            other = quaxtensor_from_scalar(other, self)
            scalar_name = f"{op_name}-scalar"
            #need to store the scalar for later still
            store_quantized(enrolled_model(), scalar_name, other)
            #TODO -need to store a quax quantized
            quaxpr_multiarg(self, other, op=op_type, mdl=enrolled_model(), op_name = op_name, has_scalar = has_scalar)
        else:
            raise TypeError(f"Unsupported type for operation: {type(other)}")

        #unquantized output
        qx = quantize(new_x, qx=self)
        #this is the hard part for me
        #need to store 
        store_activation(enrolled_model(), op_name, qx)
        quaxpr_default(qx, op_type, enrolled_model(), op_name = op_name)
        return qx

def stack(arrays, axis, requant=True):
    arrays = [expand_dims(x, axis) for x in arrays]
    arrays= concatenate(arrays, axis, requant = requant)
    return arrays

def expand_dims(x, axis):
    cur_shape = x.shape
    new_shape = list(cur_shape)
    if axis < 0:
        axis = len(new_shape) - axis + 1
    new_shape.insert(axis,1)
    new_shape = tuple(new_shape)
    return reshape(new_shape, x)

def concatenate(arrays, axis=0, requant=True):
    assert all(isinstance(x,QuaxTensor) for x in arrays), "need to use quax arrays"
    op = Operation.CONCATENATE
    mdl = enrolled_model()
    op_name = mdl.scope.default_name("op") 
    mdl.scope.reserve(op_name)
    op_name = f"element-{op_name}"
    #need to check for quantization being equal
    scales = [x.scale for x in arrays]
    #TODO - how to requantize things using the same quantization
    #let this go first because we might insert primitives to requantize
    #if not all(jnp.equal(scales[0][0], x[0]) for x in scales):
    #    arrays = [requantize(x) for x in arrays]
    #the requantizing here is messy..
    def nearly_equal(x, y):
        return jnp.array(jnp.abs(x[0] - y[0]) < .0001)
    if not requant:
        #checkify.check(jnp.all(jnp.array([nearly_equal(scales[0],x) for x in scales])), "scales aren't equal in requant")
        assert all([scales == x for x in scales])
    if requant:
        arrays = [arrays[0]] + [requantize(x, scales[0]) for x in arrays[1:]]

    for idx in range(len(arrays)):
        var_name = op_name + f"-{idx}"
        store_activation(mdl, var_name, arrays[0], scale_only=True)
    quaxpr_multiarg(*arrays, op=op, mdl=mdl, axis = axis, op_name=op_name)
    base_arrs = [x.x for x in arrays]
    qval_arrs = [x.qx.qvalue for x in arrays]

    x = jnp.concatenate(base_arrs, axis=axis)
    qval = jnp.concatenate(qval_arrs, axis=axis)
    tmp = arrays[0]

    qx = QuaxTensor(x = x, qx = tmp)   
    quaxpr_default(qx, op, mdl, axis=axis, op_name=op_name)
    store_activation(mdl, op_name, qx)

    return qx

def requantize(qx, scale, zp):
    op_type = Operation.QUANTIZE

    #TODO - how to capture calibration_axes
    op_name = enrolled_model().scope.default_name("op") 
    enrolled_model().scope.reserve(op_name)
    op_name = f"element-{op_name}"
    quaxpr_default(qx, op_type, enrolled_model(), op_name = op_name)

    qx = quantize(qx.x, calibration_axes=[x for x in range(0, qx.ndim)],
                  bits = qx.bits,
                  calibrator = quantizer.PassthroughCalibrator(),
                  po2_scaling = qx.po2_scale, scale = scale, zero_point = zp)

    mdl = enrolled_model()
    store_activation(mdl, op_name, qx)
    quaxpr_default(qx, op_type, mdl, op_name = op_name)
    return qx

def quaxtensor_transpose(x, axes=None):
    op_type = Operation.TRANSPOSE

    quaxpr_functional(x, op_type)

    # Create deep copy of original object
    new_quaxt = copy.deepcopy(x)
    
    # Modify the copy's values
    # TODO - scale will not normally change due to activation quantization strategy 
    new_quaxt.x = jnp.transpose(x.x, axes)
    #TODO - transpose scale correctly
    

    mdl = enrolled_model()
    quaxpr_default(new_quaxt, op_type, mdl, axes=axes)
    return new_quaxt


def quaxtensor_squeeze(x,axis=None):
    assert isinstance(x, QuaxTensor), "use quaxtensor"

    shape = list(x.shape)
    
    if axis is None:
        # Remove all singleton dimensions
        shape = [dim for dim in shape if dim != 1]
    else:
        # Ensure axis is a tuple
        if isinstance(axis, int):
            axis = (axis,)
        
        # Validate axes
        for ax in axis:
            if shape[ax] != 1:
                raise ValueError(f"Cannot squeeze dimension {ax}, size is {shape[ax]}, not 1.")
        
        # Remove specified singleton dimensions
        shape = [dim for i, dim in enumerate(shape) if i not in axis or dim != 1]
    return quaxtensor_reshape(shape, x)

def quaxtensor_reshape(shape, quaxt):
    quaxpr_functional(quaxt, Operation.RESHAPE )
    quaxt.x = quaxt.x.reshape(shape)
    quaxt.scale = quaxt.scale.reshape((1,) + (1,) * (quaxt.ndim-1))
    #TODO - quax reshape
    #assert quaxt.qx.scale_t is None, "don't know how to handle this"
    quaxpr_functional(quaxt, Operation.RESHAPE )
    return quaxt

def reshape(shape, x):
    return x.reshape(shape)

def sigmoid(x, out_bits):
    x = Activation(out_bits, nn.sigmoid, ActivationType.SIGMOID, scale = 1/32768., zp = 0)(x) 
    return x 

def tanh(x, out_bits):
    x = Activation(out_bits, nn.tanh, ActivationType.TANH, scale=1/32768., zp = 0)(x) 
    return x 

class Activation(nn.Module):
    lhs_bits: int
    act_fn: nn.activation
    act_type: ActivationType
    op_type: Operation = Operation.ACTIVATION
    scale: float = None
    zp: float = None

    @nn.compact
    def __call__(self, x):


        quaxpr_default(x, self.op_type,self, act_type = self.act_type )
        x = self.act_fn(x.x)
            
        #TODO - inheriting scale
        if self.scale:
            assert self.zp is not None, "must set zp if setting scale"
            qx = quantize(x, calibration_axes=[x for x in range(0, x.ndim)],
                          bits = self.bits,
                          calibrator = quantizer.PassthroughCalibrator(),
                          po2_scaling = False, scale = self.scale, zero_point = self.zp)
        else:
            qx = quantize(x, calibration_axes=[x for x in range(0, x.ndim)],
                          bits = self.bits,
                          calibrator = quantizer.AbsMaxCalibrator(),
                          po2_scaling = False, scale = self.scale, zero_point = self.zp)



        store_activation(self, 'output', qx)
        quaxpr_default(qx, self.op_type,self, act_type = self.act_type)
        return qx

def map_appended_activation(act_fn):
    act_map = {}
    act_map[nn.relu] = AppendedActivation.RELU
    act_map[nn.relu6] = AppendedActivation.RELU6
    act_map[None] = None
    return act_map[act_fn]

class GRUCell(nn.Module):
    lhs_bits: int
    rhs_bits: int
    gate_fn = sigmoid
    activation_fn = tanh
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()


    @nn.compact
    def __call__(self, carry, inputs):
        #need to wrap operation in the quaxpr
        h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
          QDense,
          features=hidden_features,
          lhs_bits = self.lhs_bits,
          rhs_bits = self.rhs_bits,
          use_bias=False,
          param_dtype=self.param_dtype,
          kernel_init=self.recurrent_kernel_init,
          bias_init=self.bias_init,
        )
        dense_i = partial(
          QDense,
          features=hidden_features,
          lhs_bits = self.lhs_bits,
          rhs_bits = self.rhs_bits,
          use_bias=True,
          param_dtype=self.param_dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
        )
        r = sigmoid(dense_i(name='ir')(inputs) + dense_h(name='hr')(h), self.lhs_bits)
        z = sigmoid(dense_i(name='iz')(inputs) + dense_h(name='hz')(h), self.lhs_bits)
        # add bias because the linear transformations aren't directly summed.
        n = tanh(
          dense_i(name='in')(inputs) + r * dense_h(name='hn', use_bias=True)(h), self.lhs_bits
        )
        #new_h = (1.0 - z) * n + z * h
        new_h = (1.0 - z) * n + z * h

        return new_h, new_h

class QDense(nn.Module):
    features: int
    lhs_bits: int
    rhs_bits: int
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    op_type: Operation = Operation.FC
    act_fn: nn.activation = None 

    @nn.compact
    def __call__(self, x):
        #need to wrap operation in the quaxpr
        quaxpr_default(x, self.op_type,self )
        store_quantized(self, 'input', x)

        allowed_acts = [nn.relu, nn.relu6, None]
        assert self.act_fn in allowed_acts, f"can't fuse act fn {self.act_fn}"
        #TODO - assignment means here

        #x = x.dequant()
        kernel = self.param(
          'kernel',
          self.kernel_init,
          (x.shape[-1], self.features),
          self.param_dtype,
        )

        kernel = quantize(kernel, calibration_axes=[_ for _ in range(0, kernel.ndim)],
                      bits = self.rhs_bits,
                      calibrator = quantizer.AbsMaxCalibrator(use_zp=False),
                      po2_scaling = False)


        store_quantized(self, 'kernel', kernel)

        if self.use_bias:
          bias = self.param(
            'bias', self.bias_init, (self.features,), self.param_dtype
          )
          #TODO - bias bits
          bias = quantize(bias, calibration_axes=[_ for _ in range(0, bias.ndim)],
                        bits = 16,
                        calibrator = quantizer.AbsMaxCalibrator(use_zp=False),
                        po2_scaling = False)
          store_quantized(self, 'bias', bias)
        else:
          bias = None
        #TODO -look at speedups from quantized vectors 
        y = lax.dot_general(
          x.x,
          kernel.x,
          (((x.ndim - 1,), (0,)), ((), ())),
          precision=None,
        )
        if bias is not None:
            #bias has been quantized and dequantized by this point 
            y += jnp.reshape(bias.x, (1,) * (x.ndim - 1) + (-1,))
            #x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))
        if self.act_fn:
            y = self.act_fn(y)
        
        qy = quantize(y, calibration_axes=[_ for _ in range(0, y.ndim)],
                      bits = self.lhs_bits,
                      calibrator = quantizer.AbsMaxCalibrator(),
                      po2_scaling = False)

        store_activation(self, 'output', qy)
        quaxpr_default(qy, self.op_type,self, act_fn = map_appended_activation(self.act_fn))
        return qy

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
  act_fn: nn.activation = None 
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
    quaxpr_default(x, self.op_type,self )
    allowed_acts = [nn.relu, nn.relu6, None]
    assert self.act_fn in allowed_acts, f"can't fuse act fn {self.act_fn}"
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



    kernel = quantize(kernel, calibration_axes=[_ for _ in range(0, kernel.ndim-1)],
                  bits = self.rhs_bits,
                  calibrator = AbsMaxCalibrator(use_zp =False),
                  po2_scaling = False)
    store_quantized(self, 'kernel', kernel)

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
      bias = quantize(bias, calibration_axes=[_ for _ in range(0, bias.ndim-1)],
                    bits = 16,
                    calibrator = quantizer.AbsMaxCalibrator(use_zp=False),
                    po2_scaling = False)

      store_quantized(self, 'bias', bias)
    else:
      bias = None


    store_quantized(self, 'input', x)
    #inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      y = lax.conv_general_dilated(
        x.x,
        kernel.x,
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
        y += bias.x.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore

    if self.act_fn:
        y = self.act_fn(y)

        
    qy = quantize(y, calibration_axes=[_ for _ in range(0, y.ndim)],
                  bits = self.lhs_bits,
                  calibrator = quantizer.AbsMaxCalibrator(),
                  po2_scaling = False)

    store_activation(self, 'output', qy)

    quaxpr_default(qy, self.op_type,self, lhs_dilation=input_dilation, rhs_dilation=kernel_dilation,window_strides=strides, padding = self.padding, act_fn = map_appended_activation(self.act_fn))
    return qy
