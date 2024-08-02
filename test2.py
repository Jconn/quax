import flatbuffers
import numpy as np

# Assuming the flatbuffer schema files have been generated and are available in your path
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType)
from quax.tflite_utils import *

builder = flatbuffers.Builder(1024)
input_data = np.zeros((1, 20), dtype=np.float32)
weights_data = np.random.rand(10,20).astype(np.float32)
biases_data = np.random.rand(1,10).astype(np.float32)
buffers = []
tensors = []
buffers.append(get_empty_buffer(builder))
tensors.append(add_empty_tensor(builder, "input", input_data.shape, buffers))
tensors.append(add_tensor(builder, "weights", weights_data, buffers) )
tensors.append(add_tensor(builder, "bias", biases_data, buffers) )
tensors.append(add_empty_tensor(builder, "output", (1,10), buffers) )
ops = []
op, opcode = add_fc_layer(builder,input_tensor=tensors[0], weight_tensor=tensors[1],bias_tensor=tensors[2] ,output_tensor=tensors[3], all_tensors=tensors) 
ops.append(op)
opcodes = [opcode]

subgraphs = []
subgraphs.append(create_subgraph(builder, tensors, [tensors[0]], [tensors[3]], ops))

model = create_model(builder, subgraphs, opcodes, buffers)
# Save the TFLite model to a file
with open('manual_model.tflite', 'wb') as f:
    f.write(export_tflite(builder,model) )
