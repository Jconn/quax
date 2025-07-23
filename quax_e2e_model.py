# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mnist example."""
import os
import argparse
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1" 
import copy
import functools
import sys
from typing import Any, Callable, Optional
from absl import app
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2.aqt_conv_general import make_conv_general_dilated, conv_general_dilated_make
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import utils
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax import aqt_flax_calibration
from flax import linen as nn
from flax import struct
from flax.metrics import tensorboard
import jax
from jax.lax import conv_dimension_numbers
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from typing import Sequence
from jax.experimental import jax2tf
import tensorflow as tf

from quax.jax2tflite import FBB
from quax.quax import QDense, Quantize, QConv, Dequantize, GRUCell
from quax import quax
from quax.quax import QModule
import orbax.checkpoint as ocp
from pathlib import Path

class CNN(QModule):
    @nn.compact
    def __call__(self, x, recurrent):
        act_bits =8
        weight_bits = 8
        bias = False 
        x = Quantize(bits=act_bits)(x)
        #TODO - why does this go unstable at stride (1,1)
        x = QConv(features=8, strides=(1,2), kernel_size=(1,3), lhs_bits = act_bits, rhs_bits = weight_bits, use_bias = True, padding='VALID')(x)
        x = x.transpose([0,2,1,3])
        #x = x + x
        x = QConv(features=10, kernel_size=(3,3), lhs_bits = act_bits, rhs_bits = weight_bits, act_fn = nn.relu, use_bias = True, padding='VALID')(x)
        x = x.reshape((x.shape[0], -1))
        x = x + x
        x = QDense(features=40,lhs_bits = act_bits, rhs_bits = weight_bits, act_fn = nn.relu, use_bias = bias)(x)
        x2 = QDense(features=40,lhs_bits = act_bits, rhs_bits = weight_bits, act_fn = nn.relu, use_bias = bias)(x)
        x,_ = GRUCell(lhs_bits = act_bits, rhs_bits = weight_bits)(x, x2)
        x = x - x2
        x,_ = GRUCell(lhs_bits = act_bits, rhs_bits = weight_bits)(x,x2)

        x = QDense(features=10,lhs_bits = act_bits, rhs_bits = weight_bits, use_bias = bias)(x)

        #x = QDense(features=10,lhs_bits = act_bits, rhs_bits = weight_bits, use_bias = bias, act_fn = nn.relu)(x)
        x = Dequantize()(x)
        return x, x 

def apply_model(model_params, images, labels, apply_fn):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(model):
    inf, updated_var = apply_fn(
        model,
        images, jnp.ones([images.shape[0],10]),
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
    logits, _ = inf
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits, updated_var)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
  aux, grads = grad_fn(model_params)
  #grads['params']['QDense_0']['kernel']
  loss, (logits, updated_var) = aux
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy, updated_var


def update_model(state, grads, updated_var):
  params = state.model['params']
  param_grad = grads['params']
  updates, new_opt_state = state.tx.update(param_grad, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)
  updated_var.update(params=new_params)
  return state.replace(
      model=updated_var,
      opt_state=new_opt_state,
  )


def _prepare_data_perm(ds, batch_size, rng, num_steps=sys.maxsize):
  ds_size = len(ds['image'])
  num_steps = min(num_steps, ds_size // batch_size)

  perms = jax.random.permutation(rng, len(ds['image']))
  perms = perms[: num_steps * batch_size]  # skip incomplete batch
  perms = perms.reshape((num_steps, batch_size))

  return perms


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  perms = _prepare_data_perm(train_ds, batch_size, rng)

  epoch_loss = []
  epoch_accuracy = []
  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy, updated_var = apply_model(
        state.model, batch_images, batch_labels, state.cnn_train.apply
    )
    state = update_model(state, grads, updated_var)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
    #jc - test early exit
    #break

  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)


  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  print('get_datasets started')
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  print('get_datasets DONE')
  return train_ds, test_ds


class TrainState(struct.PyTreeNode):
  """Train state."""

  cnn_train: Any = struct.field(pytree_node=False)
  cnn_eval: Any = struct.field(pytree_node=False)
  model: Any = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)


def create_train_state(rng):
  """Creates initial `TrainState`."""
  cnn_train = CNN(train_quant=True)
  batch_size = 128

  with jax.check_tracer_leaks():
      model = cnn_train.init({'params': rng}, jnp.ones([batch_size, 28, 28, 1]), jnp.ones([1,10]))
  learning_rate = 0.1
  momentum = 0.9
  tx = optax.sgd(learning_rate, momentum)

  def mask_fn(tree):
      return {k: (True if k == "params" else False) for k, v in tree.items()}
  #tx = optax.masked(tx, mask_fn)
  cnn_eval = CNN(train_quant=True)
  return TrainState(
      cnn_train=cnn_train,
      cnn_eval=cnn_eval,
      model=model,
      tx=tx,
      opt_state=tx.init(model['params']),
  )


def train_and_evaluate(
    num_epochs: int,
    workdir: str,
    aqt_cfg: aqt_config.DotGeneral | None = None,
    state: TrainState | None = None,
) -> TrainState:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  summary_writer = tensorboard.SummaryWriter(workdir)

  rng, init_rng = jax.random.split(rng)
  if state is None:
    state = create_train_state(init_rng)

  batch_size = 128
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)

    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, batch_size, input_rng
    )
    #_, test_loss, test_accuracy, _ = apply_model(
    #    state.model, test_ds['image'], test_ds['label'], state.cnn_eval.apply
    #)
    test_loss = 0.
    test_accuracy = 0.

    print(
        'epoch:% 3d, train_loss: %.30f, train_accuracy: %.30f, test_loss:'
        ' %.30f, test_accuracy: %.30f'
        % (
            epoch,
            train_loss,
            train_accuracy * 100,
            test_loss,
            test_accuracy * 100,
        ),
        flush=True,
    )

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state


def serving_conversion(
    train_state: TrainState,
    weight_only: bool = True,
    legacy_for_freeze: bool = False,
    legacy_for_serve: bool = False
) -> tuple[Callable[..., Any], dict[str, Any]]:
  """Model conversion (quantized weights freezing).

  Convert the model parameter for serving. During conversion, quantized weights
  are created as variables, along with their scales.

  Args:
    train_state: TrainState containing model definitions and params.
    weight_only: If set, does not quantize activations.
    legacy_for_freeze: If set, use legacy freezer during freeze.
    legacy_for_serve: If set, use legacy freezer during serve.

  Returns:
    A tuple of serving function, and converted model parameters.
  """
  aqt_cfg = train_state.cnn_eval.aqt_cfg
  input_shape = (1, 28, 28, 1)
  activation_quant_mode = (
      utils.QuantMode.TRAIN if weight_only else utils.QuantMode.SERVE
  )
  cnn_freeze = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      weights_quant_mode=utils.QuantMode.CONVERT,
      activation_quant_mode=activation_quant_mode,
      use_legacy_freezer=legacy_for_freeze
  )
  _, model_serving = cnn_freeze.apply(
      train_state.model,
      jnp.ones(input_shape),
      rngs={'params': jax.random.PRNGKey(0)},
      mutable=True,
  )
  cnn_serve = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      weights_quant_mode=utils.QuantMode.SERVE,
      activation_quant_mode=activation_quant_mode,
      use_legacy_freezer=legacy_for_serve
  )

  return cnn_serve.apply, model_serving


def _merge_pytrees(from_model, to_model):
  """Copies the parameters from from_model to to_model."""
  from_model_flattened, _ = jax.tree_util.tree_flatten_with_path(from_model)
  to_model_flattened, to_model_treedef = jax.tree_util.tree_flatten_with_path(
      to_model
  )
  from_model_kp_to_val = {kp: val for kp, val in from_model_flattened}
  merged_flattened = []
  for kp, val in to_model_flattened:
    if kp in from_model_kp_to_val:
      merged_flattened.append((kp, from_model_kp_to_val[kp]))
    else:
      merged_flattened.append((kp, val))

  merged_model = jax.tree_util.tree_unflatten(
      to_model_treedef, [v for _, v in merged_flattened]
  )

  return merged_model


def update_cfg_with_calibration(aqt_cfg):
  """Updates aqt_cfg for static range calibration."""
  sr_calibration_cls = functools.partial(
      aqt_flax_calibration.MeanOfAbsMaxCalibration,
      quant_collection='qc',
  )

  aqt_config.set_fwd_calibration(aqt_cfg, sr_calibration_cls)

  # For static range calibration, the calibration axis for activation should
  # be set to per_tensor, since its dimensions could be different during
  # training and during inference.
  aqt_cfg.fwd.dg_quantizer.lhs.calib_shared_axes = 'per_tensor'


def calibration_conversion(
    train_state: TrainState,
) -> tuple[Callable[..., Any], dict[str, Any]]:
  """Model conversion (initializing calibration parameters).

  Newly initialize variables to store the quantization statistics collected
  during calibration process.

  Args:
    train_state: TrainState containing model definitions and params.
  Returns:
    A tuple of calibration function, and an updated model parameters with new
    variables to store the gathered quantization statistics.
  """
  cnn_calibrate = CNN(
      bn_use_stats=False,
      aqt_cfg=train_state.cnn_eval.aqt_cfg,
      # Both side should be calibrated.
      weights_quant_mode=utils.QuantMode.CALIBRATE,
      activation_quant_mode=utils.QuantMode.CALIBRATE,
  )

  # Initialize the model, and then load the checkpoint into the initialized
  # parameter dict.
  model_calibrated_init = cnn_calibrate.init(
      jax.random.PRNGKey(0), jnp.ones([1, 28, 28, 1])
  )
  model_calibrated = _merge_pytrees(train_state.model, model_calibrated_init)

  return cnn_calibrate.apply, model_calibrated


def calibrate_epoch(
    calibrate_func,
    model_calibrated,
    train_ds,
    batch_size,
    rng,
    calibration_steps,
):
  """Calibrates for a single epoch."""
  perms = _prepare_data_perm(train_ds, batch_size, rng, calibration_steps)

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]

    # Calibration simply updates model during inference; it does NOT apply any
    # gradients.
    _, _, _, model_calibrated = apply_model(
        model_calibrated, batch_images, batch_labels, calibrate_func
    )

  return model_calibrated


def calibrate(state: TrainState, calibration_steps: int) -> TrainState:
  """Calibrate."""
  train_ds, _ = get_datasets()
  rng = jax.random.key(0)
  batch_size = 128
  calibration_func, model_calibrated = calibration_conversion(state)

  model_calibrated = calibrate_epoch(
      calibration_func,
      model_calibrated,
      train_ds,
      batch_size,
      rng,
      calibration_steps,
  )

  return state.replace(model=model_calibrated)


def test_translate(tflite_model, state, weight_only: bool = True):
  # get sample serving data
  _, test_ds = get_datasets()
  sample_image, sample_label = test_ds['image'][:64], test_ds['label'][:64]
  # serving
  # make tflite

  interpreter = tf.lite.Interpreter(model_path=tflite_model)
  interpreter.allocate_tensors()
  out_logs = None
  for i in range(sample_image.shape[0]):
      logits, rnn = tflite_invoke(interpreter, sample_image[i:i+1,...])
      if out_logs is None:
          out_logs = logits
      else:
          out_logs = jnp.concat([out_logs, logits], axis=0)

  # compute serving loss
  one_hot = jax.nn.one_hot(sample_label, 10)

  inf, updated_var = state.cnn_train.apply(state.model,sample_image,jnp.ones([1,10]), rngs={'params': jax.random.PRNGKey(0)},mutable=True,)
  mlogits, jax_rnn = inf
  tfl_loss = jnp.mean(optax.softmax_cross_entropy(logits=out_logs, labels=one_hot))
  jax_loss = jnp.mean(optax.softmax_cross_entropy(logits=mlogits, labels=one_hot))
  if jnp.abs(jax_loss - tfl_loss) > .05:
      raise Exception(f"tflite jax loss differs - tfl; {tfl_loss}, jax - {jax_loss}")
  else:
      print(f"tflite jax PASS - tfl; {tfl_loss}, jax - {jax_loss}")

def tflite_invoke(interpreter, float_input):

    # Get input and output details

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']




    # Read quantization parameters
    quantization_params = input_details[0]['quantization']  # (scale, zero_point)
    scale, zero_point = quantization_params

    # Quantize the float input to int16
    #quantized_input = np.round(float_input / scale + zero_point).astype(np.int16)
    quantized_input = float_input

    # Ensure the input tensor matches the expected type and shape
    #interpreter.set_tensor(0,input_index)

    interpreter.set_tensor(input_index, quantized_input)
    # Run inference
    interpreter.invoke()
    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    rnn_output = interpreter.get_tensor(output_details[1]['index'])

    # If the output is quantized, dequantize it for interpretation
    output_quant_params = output_details[0]['quantization']
    output_scale, output_zero_point = output_quant_params
    if output_scale != 0:  # Check if quantization is used
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    return output_data, rnn_output

def main():
    # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--load_weights', type=str, default=None, help='Path to model weights to load (Orbax checkpoint directory)')
  args = parser.parse_args()

  ckpt_root   = Path("checkpoints")
  ckpt_root.mkdir(parents=True, exist_ok=True)
  step = 1 # use a single tag here it's not important
  while (ckpt_root / str(step)).exists():
      step += 1
  ckpt_path = ckpt_root / str(step)
  ckpt_path = ckpt_path.resolve()
  checkpointer = ocp.StandardCheckpointer()

  # 1. TRAIN or LOAD.
  if args.load_weights is not None:
      print(f"Loading model weights from {args.load_weights} using Orbax")
      ckpt_path = Path(args.load_weights).resolve()
      save_state = create_train_state(jax.random.key(0)) 
      abstract = jax.tree_map(ocp.utils.to_shape_dtype_struct, save_state)
      state = checkpointer.restore(ckpt_path, target=abstract)
  else:
      state = train_and_evaluate(
              num_epochs=3, workdir='/tmp/aqt_mnist_example')
      # Save weights using Orbax after training

      checkpointer.save(ckpt_path, state)
      print("saved orbax ckpt")

  x = jnp.ones([1, 28, 28, 1])
  converter = FBB()
  with jax.disable_jit():
      tflite = converter.convert(state.cnn_train, state.model, x=x, recurrent=jnp.ones([1,10]))
  with open('mnist_quax.tflite', 'wb') as f:
      f.write(tflite)

  cnn_train = CNN(train_quant=False)
  state = state.replace(cnn_train=cnn_train)
  test_translate('mnist_quax.tflite', state, weight_only=False)

  #(Pdb) serving_model['aqt']['Conv_1']['AqtConvGeneralDilated_0']['qrhs']
if __name__ == '__main__':
    with jax.disable_jit():
        main()
