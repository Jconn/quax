import flatbuffers
import numpy as np
# Assuming the flatbuffer schema files have been generated and are available in your path
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType)

def add_empty_tensor(builder, tensor_name, tensor_dims, buffers):
    #TODO - datatype resolution
    tensor_name = builder.CreateString(tensor_name)
    Tensor.StartShapeVector(builder, len(tensor_dims))
    for dim in reversed(tensor_dims):
        builder.PrependInt32(dim)
    tensor_shape = builder.EndVector()
    

    #create tensor
    Tensor.Start(builder)
    Tensor.AddShape(builder, tensor_shape)
    #TODO - conversion from np data to type
    Tensor.AddType(builder, 0)  # float32
    #the 0 buffer is standard notation for empty buffer (e.g. activations)
    Tensor.AddBuffer(builder, 0) 
    Tensor.AddName(builder, tensor_name)
    tensor = Tensor.End(builder)
    return tensor

#tensor add should have a numpy and string as input
def add_tensor(builder, tensor_name, np_data, buffers):
    #TODO - datatype resolution
    np_shape = np_data.shape
    tensor_name = builder.CreateString(tensor_name)

    Tensor.StartShapeVector(builder, len(np_shape))
    for dim in reversed(np_shape):
        builder.PrependInt32(dim)
    tensor_shape = builder.EndVector()
    
    #TODO - need to deal with empty buffer
    #add buffer we use here
    data_vector = builder.CreateByteVector(np.ravel(np_data).tobytes())
    Buffer.Start(builder)
    Buffer.AddData(builder, data_vector)
    buffer = Buffer.End(builder)
    buffers.append(buffer)
    buffer_idx = buffers.index(buffer)

    #create tensor
    Tensor.Start(builder)
    Tensor.AddShape(builder, tensor_shape)
    #TODO - conversion from np data to type
    Tensor.AddType(builder, 0)  # float32
    Tensor.AddBuffer(builder, buffer_idx) 
    Tensor.AddName(builder, tensor_name)
    tensor = Tensor.End(builder)
    return tensor

def get_empty_buffer(builder):
    Buffer.Start(builder)
    empty_buffer = Buffer.End(builder)
    return empty_buffer

def add_buffer(builder, buffers, data = None):
    Buffer.Start(builder)
    if data is not None:
        data_vector = builder.CreateByteVector(np.ravel(data).tobytes())
        Buffer.AddData(builder, data_vector)
    buffer = Buffer.End(builder)
    buffers.append(buffer)
    return buffer

def add_fc_layer(builder, input_tensor, weight_tensor, bias_tensor, output_tensor, all_tensors):
    # Create the FullyConnectedOptions
    FullyConnectedOptions.Start(builder)
    #TODO - need to deal with fusion here - is here a way to do this?
    FullyConnectedOptions.AddFusedActivationFunction(builder, ActivationFunctionType.ActivationFunctionType().RELU)
    fc_options = FullyConnectedOptions.End(builder)
    # Create the OperatorCode for FullyConnected
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().FULLY_CONNECTED)
    fc_op_code = OperatorCode.End(builder)
    
    #TODO - ordering here is fragile but important
    fc_inputs = create_operator_inputs(builder, [input_tensor, weight_tensor, bias_tensor], all_tensors)
    fc_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, 0)
    Operator.AddInputs(builder, fc_inputs)
    Operator.AddOutputs(builder, fc_outputs)
    Operator.AddBuiltinOptions(builder, fc_options)
    Operator.AddBuiltinOptionsType(builder, BuiltinOptions.BuiltinOptions().FullyConnectedOptions)
    fc_op = Operator.End(builder)
    return fc_op, fc_op_code

def create_subgraph_tensors(builder, tensors):
    SubGraph.StartTensorsVector(builder, len(tensors))
    for tensor in reversed(tensors):
        builder.PrependUOffsetTRelative(tensor)
    subgraph_tensors = builder.EndVector()
    return subgraph_tensors

def create_operator_inputs(builder, input_tensors, all_tensors):
    Operator.StartInputsVector(builder, len(input_tensors))
    for itensor in reversed(input_tensors):
        builder.PrependInt32(all_tensors.index(itensor))  # input tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_operator_outputs(builder, output_tensors, all_tensors):
    Operator.StartOutputsVector(builder, len(output_tensors))
    for otensor in reversed(output_tensors):
        builder.PrependInt32(all_tensors.index(otensor))  # input tensor index
    subgraph_outputs = builder.EndVector()
    return subgraph_outputs


def create_subgraph_inputs(builder, input_tensors, all_tensors):
    SubGraph.StartInputsVector(builder, 1)
    for itensor in reversed(input_tensors):
        builder.PrependInt32(all_tensors.index(itensor))  # input tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_subgraph_outputs(builder, output_tensors, all_tensors):
    SubGraph.StartOutputsVector(builder, len(output_tensors))
    for otensor in reversed(output_tensors):
        builder.PrependInt32(all_tensors.index(otensor))  # output tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_subgraph_ops(builder, ops):
    SubGraph.StartOperatorsVector(builder, len(ops))
    for op in reversed(ops):
        builder.PrependUOffsetTRelative(op)
    subgraph_ops = builder.EndVector()
    return subgraph_ops

def create_subgraph(builder, subgraph_tensors, subgraph_inputs, subgraph_outputs, subgraph_ops):

    #formalize builder constructs from python lists
    subgraph_inputs = create_subgraph_inputs(builder, subgraph_inputs, subgraph_tensors)
    subgraph_outputs = create_subgraph_outputs(builder, subgraph_outputs, subgraph_tensors)
    subgraph_tensors = create_subgraph_tensors(builder, subgraph_tensors)
    subgraph_ops = create_subgraph_ops(builder, subgraph_ops)

    SubGraph.Start(builder)
    SubGraph.AddTensors(builder, subgraph_tensors)
    SubGraph.AddInputs(builder, subgraph_inputs)
    SubGraph.AddOutputs(builder, subgraph_outputs)
    SubGraph.AddOperators(builder, subgraph_ops)
    subgraph = SubGraph.End(builder)
    return subgraph

def create_model_subgraphs(builder, subgraphs):
    Model.StartSubgraphsVector(builder, len(subgraphs))
    for subgraph in reversed(subgraphs):
        builder.PrependUOffsetTRelative(subgraph)
    subgraphs = builder.EndVector()
    return subgraphs

def create_opcodes(builder, op_codes):
    Model.StartOperatorCodesVector(builder, len(op_codes))
    for opcode in reversed(op_codes):
        builder.PrependUOffsetTRelative(opcode)
    op_codes = builder.EndVector()
    return op_codes

def create_buffers(builder, buffers):
    Model.StartBuffersVector(builder, len(buffers))
    for buffer in reversed(buffers):
        builder.PrependUOffsetTRelative(buffer)
    buffers = builder.EndVector()
    return buffers

def create_model(builder, subgraphs, op_codes, buffers):
    subgraphs = create_model_subgraphs(builder, subgraphs)
    op_codes = create_opcodes(builder, op_codes)
    buffers = create_buffers(builder, buffers)

    Model.Start(builder)
    Model.AddVersion(builder, 3)
    Model.AddSubgraphs(builder, subgraphs)
    Model.AddOperatorCodes(builder, op_codes)
    Model.AddBuffers(builder, buffers)
    model = Model.End(builder)
    return model

def export_tflite(builder, model):
    builder.Finish(model)
    tflite_model_data = builder.Output()
    return tflite_model_data
