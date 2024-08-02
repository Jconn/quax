
import flatbuffers
import numpy as np

# Assuming the flatbuffer schema files have been generated and are available in your path
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType)

# Create a flatbuffer builder
builder = flatbuffers.Builder(1024)

# Create the input data
input_data = np.zeros((1, 20), dtype=np.float32)
input_data_vector = builder.CreateByteVector(np.ravel(input_data).tobytes())

# Create weights buffer data
weights_data = np.random.rand(10,20).astype(np.float32)
#create numpy vector appears to be bugged
weights_data_vector = builder.CreateByteVector(np.ravel(weights_data).tobytes())

# Create biases buffer data
biases_data = np.random.rand(1,10).astype(np.float32)
#biases_data_vector = builder.CreateNumpyVector(np.ravel(biases_data))
biases_data_vector = builder.CreateByteVector(np.ravel(biases_data).tobytes())
print(biases_data)
# Create an empty buffer for the output tensor
Buffer.Start(builder)
empty_buffer = Buffer.End(builder)

# Create input buffer
Buffer.Start(builder)
Buffer.AddData(builder, input_data_vector)
input_buffer = Buffer.End(builder)

# Create weights buffer
Buffer.Start(builder)
Buffer.AddData(builder, weights_data_vector)
weights_buffer = Buffer.End(builder)

# Create biases buffer
Buffer.Start(builder)
Buffer.AddData(builder, biases_data_vector)
biases_buffer = Buffer.End(builder)

# Create the input tensor
input_tensor_name = builder.CreateString("input_tensor")
Tensor.StartShapeVector(builder, 2)
builder.PrependInt32(20)
builder.PrependInt32(1)
input_shape = builder.EndVector()


Tensor.Start(builder)
Tensor.AddShape(builder, input_shape)
Tensor.AddType(builder, 0)  # float32
Tensor.AddBuffer(builder, 3)  # The input buffer index
Tensor.AddName(builder, input_tensor_name)
input_tensor = Tensor.End(builder)

# Create the output tensor
output_tensor_name = builder.CreateString("output_tensor")
Tensor.StartShapeVector(builder, 2)
builder.PrependInt32(10)
builder.PrependInt32(1)
output_shape = builder.EndVector()

Tensor.Start(builder)
Tensor.AddShape(builder, output_shape)
Tensor.AddType(builder, 0)  # float32
Tensor.AddBuffer(builder, 3)  # The empty buffer index
Tensor.AddName(builder, output_tensor_name)
output_tensor = Tensor.End(builder)

# Create the weights tensor
weights_tensor_name = builder.CreateString("weights_tensor")
Tensor.StartShapeVector(builder, 2)
builder.PrependInt32(20)
builder.PrependInt32(10)
weights_shape = builder.EndVector()

Tensor.Start(builder)
Tensor.AddShape(builder, weights_shape)
Tensor.AddType(builder, 0)  # float32
Tensor.AddBuffer(builder, 1)  # The weights buffer index
Tensor.AddName(builder, weights_tensor_name)
weights_tensor = Tensor.End(builder)

# Create the biases tensor
biases_tensor_name = builder.CreateString("biases_tensor")
Tensor.StartShapeVector(builder, 1)
builder.PrependInt32(10)
biases_shape = builder.EndVector()

Tensor.Start(builder)
Tensor.AddShape(builder, biases_shape)
Tensor.AddType(builder, 0)  # float32
Tensor.AddBuffer(builder, 0)  # The biases buffer index
Tensor.AddName(builder, biases_tensor_name)
biases_tensor = Tensor.End(builder)

# Create the FullyConnectedOptions
FullyConnectedOptions.Start(builder)
FullyConnectedOptions.AddFusedActivationFunction(builder, ActivationFunctionType.ActivationFunctionType().RELU)
fc_options = FullyConnectedOptions.End(builder)

# Create the OperatorCode for FullyConnected
OperatorCode.Start(builder)
OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().FULLY_CONNECTED)
fc_op_code = OperatorCode.End(builder)

# Create the Operator
Operator.StartInputsVector(builder, 3)
builder.PrependInt32(0)  # biases tensor index
builder.PrependInt32(1)  # weights tensor index
builder.PrependInt32(3)  # input tensor index 
fc_inputs = builder.EndVector()

Operator.StartOutputsVector(builder, 1)
builder.PrependInt32(2)  # output tensor index
fc_outputs = builder.EndVector()

Operator.Start(builder)
Operator.AddOpcodeIndex(builder, 0)
Operator.AddInputs(builder, fc_inputs)
Operator.AddOutputs(builder, fc_outputs)
Operator.AddBuiltinOptions(builder, fc_options)
Operator.AddBuiltinOptionsType(builder, BuiltinOptions.BuiltinOptions().FullyConnectedOptions)
fc_op = Operator.End(builder)

# Create the SubGraph
SubGraph.StartTensorsVector(builder, 4)
builder.PrependUOffsetTRelative(input_tensor)
builder.PrependUOffsetTRelative(output_tensor)
builder.PrependUOffsetTRelative(weights_tensor)
builder.PrependUOffsetTRelative(biases_tensor)
subgraph_tensors = builder.EndVector()

SubGraph.StartInputsVector(builder, 1)
builder.PrependInt32(3)  # input tensor index
subgraph_inputs = builder.EndVector()

SubGraph.StartOutputsVector(builder, 1)
builder.PrependInt32(2)  # output tensor index
subgraph_outputs = builder.EndVector()

SubGraph.StartOperatorsVector(builder, 1)
builder.PrependUOffsetTRelative(fc_op)
subgraph_ops = builder.EndVector()

SubGraph.Start(builder)
SubGraph.AddTensors(builder, subgraph_tensors)
SubGraph.AddInputs(builder, subgraph_inputs)
SubGraph.AddOutputs(builder, subgraph_outputs)
SubGraph.AddOperators(builder, subgraph_ops)
subgraph = SubGraph.End(builder)

# Create the model
Model.StartSubgraphsVector(builder, 1)
builder.PrependUOffsetTRelative(subgraph)
subgraphs = builder.EndVector()

Model.StartOperatorCodesVector(builder, 1)
builder.PrependUOffsetTRelative(fc_op_code)
op_codes = builder.EndVector()

Model.StartBuffersVector(builder, 4)
builder.PrependUOffsetTRelative(empty_buffer)
builder.PrependUOffsetTRelative(input_buffer)
builder.PrependUOffsetTRelative(weights_buffer)
builder.PrependUOffsetTRelative(biases_buffer)
buffers = builder.EndVector()

Model.Start(builder)
Model.AddVersion(builder, 3)
Model.AddSubgraphs(builder, subgraphs)
Model.AddOperatorCodes(builder, op_codes)
Model.AddBuffers(builder, buffers)
model = Model.End(builder)

builder.Finish(model)
tflite_model_data = builder.Output()

# Save the TFLite model to a file
with open('manual_model.tflite', 'wb') as f:
    f.write(tflite_model_data)

print("TFLite model created successfully.")
