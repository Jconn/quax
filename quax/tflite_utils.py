import flatbuffers
import numpy as np
# Assuming the flatbuffer schema files have been generated and are available in your path
import quax.tflite as tfl
from quax.tflite import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,ConcatenationOptions,
                                        ActivationFunctionType, AddOptions, MulOptions, TensorMap,
                                        SignatureDef, Metadata, QuantizationParameters, ReshapeOptions, TensorType, Conv2DOptions,ActivationFunctionType, Padding, QuantizeOptions, StridedSliceOptions, SliceOptions, DequantizeOptions)

from enum import Enum
import jax.numpy as jnp
import numpy as np
from quax.quax import AppendedActivation
_TFLITE_FILE_IDENTIFIER = b'TFL3'

def map_appended_activation(appended_activation):
    if appended_activation == None:
        return ActivationFunctionType.ActivationFunctionType.NONE
    act_map= {}
    act_map[AppendedActivation.RELU] = ActivationFunctionType.ActivationFunctionType.RELU
    act_map[AppendedActivation.RELU6] = ActivationFunctionType.ActivationFunctionType.TANH
    return act_map[appended_activation] 

def map_tensor_type(dtype):
    dtype_map = {}
    dtype_map[np.float32] = TensorType.TensorType.FLOAT32
    dtype_map[np.int32] = TensorType.TensorType.INT32
    dtype_map[np.int8] = TensorType.TensorType.INT8
    dtype_map[np.int16] = TensorType.TensorType.INT16
    dtype_map[np.int64] = TensorType.TensorType.INT64
    return dtype_map[dtype]



def add_empty_tensor(builder, tensor_name, tensor_dims, buffers, quantization_params = None, dtype = np.float32):
    #TODO - datatype resolution
    tensor_name = builder.CreateString(tensor_name)
    Tensor.StartShapeVector(builder, len(tensor_dims))
    for dim in reversed(tensor_dims):
        builder.PrependInt32(dim)
    tensor_shape = builder.EndVector()
    
    if quantization_params is None:
        quantization_params = add_empty_quant(builder)
    #create tensor
    Tensor.Start(builder)
    Tensor.AddShape(builder, tensor_shape)
    Tensor.AddType(builder, map_tensor_type(dtype) ) 
    #the 0 buffer is standard notation for empty buffer (e.g. activations)
    Tensor.AddBuffer(builder, 0) 
    Tensor.AddName(builder, tensor_name)
    Tensor.AddQuantization(builder,quantization_params)
    tensor = Tensor.End(builder)
    return tensor

def add_quantization_params(builder, mins, maxs, scale, zero_point, quantized_dim):
    #convert all the vectors
    def to_vec(x):
        if x is not None:
            x= np.array(x)
            return builder.CreateNumpyVector(np.ravel(x) )
        return None
    zero_point = np.array(zero_point, dtype=np.int64)
    mins = to_vec(mins) 
    maxs = to_vec(maxs)
    scale = to_vec(scale) 
    zero_point = to_vec(zero_point)
    #builder.CreateByteVector(np.ravel(np.array(zero_point, dtype=np.int16)).tobytes())

    QuantizationParameters.Start(builder)
    if mins is not None:
        QuantizationParameters.AddMin(builder, mins)
    if maxs is not None:
        QuantizationParameters.AddMax(builder, maxs)

    QuantizationParameters.AddScale(builder, scale)
    QuantizationParameters.AddZeroPoint(builder, zero_point)
    QuantizationParameters.AddQuantizedDimension(builder, quantized_dim)
    QuantizationParameters.AddDetailsType(builder, 0)
    qparams = QuantizationParameters.End(builder)
    return qparams




def add_tensor_with_buffer(builder, tensor_name, np_shape, buffer, buffers, dtype = np.float32):
    #TODO - datatype resolution
    tensor_name = builder.CreateString(tensor_name)

    Tensor.StartShapeVector(builder, len(np_shape))
    #TODO - need to sort out if this really needs reversing - jaxpr standards seem to be backwards
    for dim in reversed(np_shape):
        builder.PrependInt32(dim)
    tensor_shape = builder.EndVector()
    
    #TODO - need to deal with empty buffer
    #add buffer we use here
    buffer_idx = buffers.index(buffer)

    #create tensor
    quant = add_empty_quant(builder)
    Tensor.Start(builder)
    Tensor.AddShape(builder, tensor_shape)
    Tensor.AddType(builder, map_tensor_type(dtype) ) 
    Tensor.AddBuffer(builder, buffer_idx) 
    Tensor.AddName(builder, tensor_name)
    Tensor.AddQuantization(builder,quant)
    tensor = Tensor.End(builder)
    return tensor

def add_empty_quant(builder):
    QuantizationParameters.QuantizationParametersStart(builder)
    quantization = QuantizationParameters.QuantizationParametersEnd(builder)
    return quantization

#tensor add should have a numpy and string as input
def add_tensor(builder, tensor_name, np_data, buffers, quantization_params = None, dtype = np.float32):
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
    if quantization_params is None:
        quantization_params = add_empty_quant(builder)

    Tensor.Start(builder)
    Tensor.AddShape(builder, tensor_shape)
    Tensor.AddType(builder, map_tensor_type(dtype) ) 
    Tensor.AddBuffer(builder, buffer_idx) 
    Tensor.AddName(builder, tensor_name)
    Tensor.AddQuantization(builder,quantization_params)
    tensor = Tensor.End(builder)
    return tensor

def get_empty_buffer(builder):
    Buffer.Start(builder)
    empty_buffer = Buffer.End(builder)
    return empty_buffer

def add_string_buffer(builder, buffers, str_data):
    str_data = builder.CreateString(str_data)
    Buffer.Start(builder)
    Buffer.AddData(builder, str_data)
    buffer = Buffer.End(builder)
    buffers.append(buffer)
    return buffer

def add_buffer(builder, buffers, data = None):
    #need to create everything being used in the buffer construction
    if data is not None:
        data_vector = builder.CreateByteVector(np.ravel(data).tobytes())
    Buffer.Start(builder)
    if data is not None:
        Buffer.AddData(builder, data_vector)
    buffer = Buffer.End(builder)
    buffers.append(buffer)
    return buffer

def add_operator(builder, inputs, outputs, options,options_type, opcode, all_opcodes):
    if opcode not in all_opcodes:
        all_opcodes.append(opcode)

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, all_opcodes.index(opcode))
    Operator.AddInputs(builder, inputs)
    Operator.AddOutputs(builder, outputs)
    if options:
        Operator.AddBuiltinOptions(builder, options)
    if options_type:
        Operator.AddBuiltinOptionsType(builder, options_type)
    op = Operator.End(builder)
    return op

def add_vec_layer(builder, input_tensor1, input_tensor2, output_tensor,vec_op, all_tensors, all_opcodes):
    options = None 
    vec_options = None
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, vec_op)
    add_opcode = OperatorCode.End(builder)
    # Create inputs and outputs
    inputs = create_operator_inputs(builder, [input_tensor1, input_tensor2], all_tensors)
    outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    add_op = add_operator(builder, inputs, outputs, options, vec_options, add_opcode, all_opcodes)
    return add_op

def add_concat_layer(builder, input_tensors, output_tensor, axis,all_tensors, all_opcodes):

    ConcatenationOptions.Start(builder) 
    ConcatenationOptions.AddAxis(builder, axis)
    concat_options = ConcatenationOptions.End(builder) 
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().CONCATENATION)
    concat_opcode = OperatorCode.End(builder)
    concat_inputs = create_operator_inputs(builder, input_tensors, all_tensors)
    concat_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)
    concat_op = add_operator(builder, concat_inputs, concat_outputs, concat_options, BuiltinOptions.BuiltinOptions().ConcatenationOptions, concat_opcode, all_opcodes)
    return concat_op

def add_mul_layer(builder, input_tensor1, input_tensor2, output_tensor, all_tensors, all_opcodes):
    MulOptions.Start(builder)
    mul_options = MulOptions.End(builder)
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().MUL)
    mul_opcode = OperatorCode.End(builder)
    # Create inputs and outputs
    mul_inputs = create_operator_inputs(builder, [input_tensor1, input_tensor2], all_tensors)
    mul_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    mul_op = add_operator(builder, mul_inputs, mul_outputs, mul_options, BuiltinOptions.BuiltinOptions().MulOptions, mul_opcode, all_opcodes)
    return mul_op

def add_relu_layer(builder, input_tensor, output_tensor, all_tensors, all_opcodes):
    # Create the ActivationFunctionType for ReLU

    # Create the OperatorCode for ReLU
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().RELU)
    relu_opcode = OperatorCode.End(builder)
    
    # Create inputs and outputs
    relu_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    relu_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    relu_op = add_operator(builder, relu_inputs, relu_outputs, None, None, relu_opcode, all_opcodes)
    return relu_op


def add_slice_layer(builder, input_tensor, output_tensor, slicing_key, all_tensors, all_opcodes, all_buffers):
    """
    Adds a StridedSlice operator to the TFLite flatbuffer model.
    
    Args:
        builder: FlatBufferBuilder for constructing the TFLite model.
        input_tensor: Input tensor index.
        output_tensor: Output tensor index.
        slicing_key: Dictionary specifying 'begin', 'end', and 'stride' for each dimension.
                     Example: {'begin': [0, 0, 2], 'end': [1, 193, 10], 'stride': [1, 1, 1]}
        all_tensors: List of all tensors in the model.
        all_opcodes: List of all operator codes in the model.
    Returns:
        The created StridedSlice operator.
    """
    # Extract slicing parameters
    begin = []
    end = []
    strides = []
    begin_mask = 0
    end_mask = 0

    for i, s in enumerate(slicing_key):
        if isinstance(s, slice):
            # Handle `slice(start, stop, stride)`
            begin.append(0 if s.start is None else s.start)
            end.append(0 if s.stop is None else s.stop)
            strides.append(1 if s.step is None else s.step)
            
            # Adjust masks for None (default slicing)
            if s.start is None:
                begin_mask |= (1 << i)  # Ignore the begin value for this dimension
            if s.stop is None:
                end_mask |= (1 << i)    # Ignore the end value for this dimension
        else:
            raise ValueError(f"Unsupported slicing type {type(s)} at dimension {i}")

    # Create the `begin`, `end`, and `stride` tensors 
    op_inputs = [input_tensor]
    for idx,stride_elem in enumerate((begin, end, strides)):
        dtype = np.int32
        np_arr = np.array(stride_elem, dtype=dtype)
        in_tensor = add_tensor(builder, f"stride_{idx}", np_arr, all_buffers, dtype=dtype)
        op_inputs.append(in_tensor)
        #TODO - a little messy to do things this way decentralized tensor accumulation
        all_tensors.append(in_tensor)

    # Configure the options
    StridedSliceOptions.StridedSliceOptionsStart(builder)
    StridedSliceOptions.StridedSliceOptionsAddBeginMask(builder, 0)  # Customize as needed
    StridedSliceOptions.StridedSliceOptionsAddEndMask(builder, 0)    # Customize as needed
    StridedSliceOptions.StridedSliceOptionsAddEllipsisMask(builder, 0)
    StridedSliceOptions.StridedSliceOptionsAddNewAxisMask(builder, 0)
    StridedSliceOptions.StridedSliceOptionsAddShrinkAxisMask(builder, 0)
    strided_slice_options = StridedSliceOptions.StridedSliceOptionsEnd(builder)

    # Create the OperatorCode for StridedSlice
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().STRIDED_SLICE)
    strided_slice_opcode = OperatorCode.End(builder)

    # Create inputs and outputs
    strided_slice_inputs = create_operator_inputs(builder, op_inputs, all_tensors)
    strided_slice_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    # Add the StridedSlice operator
    strided_slice_op = add_operator(builder, strided_slice_inputs, strided_slice_outputs, 
                                    strided_slice_options, BuiltinOptions.BuiltinOptions().StridedSliceOptions, 
                                    strided_slice_opcode, all_opcodes)

    return strided_slice_op

def add_transpose_layer(builder, input_tensor, output_tensor, new_shape, all_tensors, all_opcodes):
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().TRANSPOSE)
    transpose_opcode = OperatorCode.End(builder)
    
    # Create inputs and outputs
    transpose_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    transpose_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    transpose_op = add_operator(builder, transpose_inputs, transpose_outputs, None, None, transpose_opcode, all_opcodes)
    return transpose_op

def add_reshape_layer(builder, input_tensor, output_tensor, new_shape, all_tensors, all_opcodes):
    # Create the ReshapeOptions
    ReshapeOptions.StartNewShapeVector(builder, len(new_shape))
    for dim in reversed(new_shape):
        builder.PrependInt32(dim)
    new_shape_vector = builder.EndVector()

    ReshapeOptions.ReshapeOptionsStart(builder)
    ReshapeOptions.ReshapeOptionsAddNewShape(builder, new_shape_vector)
    reshape_options = ReshapeOptions.ReshapeOptionsEnd(builder)

    # Create the OperatorCode for Reshape
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape_opcode = OperatorCode.End(builder)
    
    # Create inputs and outputs
    reshape_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    reshape_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    reshape_op = add_operator(builder, reshape_inputs, reshape_outputs, reshape_options, BuiltinOptions.BuiltinOptions().ReshapeOptions, reshape_opcode, all_opcodes)
    return reshape_op

def add_conv_layer(builder, input_tensor, weight_tensor, bias_tensor, output_tensor,bias_dtype,activation_op, all_tensors, all_opcodes,quax_params):
    Conv2DOptions.Start(builder)
    #TODO - need to deal with fusion here - is here a way to do this?
    #TODO - conv stride options
    Conv2DOptions.AddStrideH(builder, 1)
    Conv2DOptions.AddStrideW(builder, 1)
    Conv2DOptions.AddFusedActivationFunction(builder,activation_op)
    if quax_params['padding'] == 'SAME':
        padding = Padding.Padding.SAME 
    else:
        padding = Padding.Padding.VALID 
    Conv2DOptions.AddPadding(builder, padding)
    conv_options = Conv2DOptions.End(builder)
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().CONV_2D)
    conv_opcode = OperatorCode.End(builder)
    
    #TODO - ordering here is fragile but important

    if bias_tensor:
        conv_inputs = create_operator_inputs(builder, [input_tensor, weight_tensor, bias_tensor], all_tensors)
    else:
        conv_inputs = create_operator_inputs(builder, [input_tensor, weight_tensor], all_tensors)

    conv_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, 0)
    Operator.AddInputs(builder, conv_inputs)
    Operator.AddOutputs(builder, conv_outputs)
    Operator.AddBuiltinOptions(builder, conv_options)
    Operator.AddBuiltinOptionsType(builder, BuiltinOptions.BuiltinOptions().Conv2DOptions)
    conv_op = Operator.End(builder)

    conv_op = add_operator(builder, conv_inputs, conv_outputs, conv_options, BuiltinOptions.BuiltinOptions().Conv2DOptions, conv_opcode, all_opcodes)
    return conv_op

def add_activation_layer(builder, input_tensor, output_tensor,operator_type, all_tensors, all_opcodes):
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, operator_type)
    act_opcode = OperatorCode.End(builder)
    act_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    act_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)
    act_op = add_operator(builder, act_inputs, act_outputs, None, None, act_opcode, all_opcodes)
    return act_op

def add_dequant_layer(builder, input_tensor, output_tensor, all_tensors, all_opcodes):
    # Create the FullyConnectedOptions
    DequantizeOptions.Start(builder)
    dequant_options = DequantizeOptions.End(builder)
    # Create the OperatorCode for FullyConnected
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().DEQUANTIZE)
    quant_opcode = OperatorCode.End(builder)
    
    #TODO - ordering here is fragile but important

    dequant_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    dequant_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    #Operator.Start(builder)
    #Operator.AddOpcodeIndex(builder, 0)
    #Operator.AddInputs(builder, dequant_inputs)
    #Operator.AddOutputs(builder, dequant_outputs)
    #Operator.AddBuiltinOptions(builder, dequant_options)
    #Operator.AddBuiltinOptionsType(builder, BuiltinOptions.BuiltinOptions().DequantizeOptions)

    dequant_op = add_operator(builder, dequant_inputs, dequant_outputs, dequant_options, BuiltinOptions.BuiltinOptions().DequantizeOptions, quant_opcode, all_opcodes)
    return dequant_op
def add_quant_layer(builder, input_tensor, output_tensor, all_tensors, all_opcodes):
    # Create the FullyConnectedOptions
    QuantizeOptions.Start(builder)
    #TODO - need to deal with fusion here - is here a way to do this?
    #FullyConnectedOptions.AddFusedActivationFunction(builder, ActivationFunctionType.ActivationFunctionType().RELU)
    quant_options = QuantizeOptions.End(builder)
    # Create the OperatorCode for FullyConnected
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().QUANTIZE)
    quant_opcode = OperatorCode.End(builder)
    
    #TODO - ordering here is fragile but important

    quant_inputs = create_operator_inputs(builder, [input_tensor], all_tensors)
    quant_outputs = create_operator_outputs(builder, [output_tensor], all_tensors)

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, 0)
    Operator.AddInputs(builder, quant_inputs)
    Operator.AddOutputs(builder, quant_outputs)
    Operator.AddBuiltinOptions(builder, quant_options)
    Operator.AddBuiltinOptionsType(builder, BuiltinOptions.BuiltinOptions().QuantizeOptions)
    quant_op = Operator.End(builder)

    quant_op = add_operator(builder, quant_inputs, quant_outputs, quant_options, BuiltinOptions.BuiltinOptions().QuantizeOptions, quant_opcode, all_opcodes)
    return quant_op

def add_fc_layer(builder, input_tensor, weight_tensor, bias_tensor, output_tensor,bias_dtype,activation_op, all_tensors, all_opcodes):
    # Create the FullyConnectedOptions
    FullyConnectedOptions.Start(builder)
    #TODO - need to deal with fusion here - is here a way to do this?
    #FullyConnectedOptions.AddFusedActivationFunction(builder, ActivationFunctionType.ActivationFunctionType().RELU)
    FullyConnectedOptions.AddFusedActivationFunction(builder,activation_op)
    fc_options = FullyConnectedOptions.End(builder)
    # Create the OperatorCode for FullyConnected
    OperatorCode.Start(builder)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.BuiltinOperator().FULLY_CONNECTED)
    fc_opcode = OperatorCode.End(builder)
    
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

    fc_op = add_operator(builder, fc_inputs, fc_outputs, fc_options, BuiltinOptions.BuiltinOptions().FullyConnectedOptions, fc_opcode, all_opcodes)
    return fc_op


def create_signature_def(builder, input_tensors, output_tensors, all_tensors, subgraph_index):
    # Create TensorMaps for inputs
    signature_key = builder.CreateString("serving_default")
    input_maps = []
    for idx, tensor in enumerate(input_tensors):
        name = f"input-{idx}"
        name_offset = builder.CreateString(name)
        TensorMap.TensorMapStart(builder)
        TensorMap.TensorMapAddName(builder, name_offset)
        TensorMap.TensorMapAddTensorIndex(builder, all_tensors.index(tensor))
        input_map = TensorMap.TensorMapEnd(builder)
        input_maps.append(input_map)

    # Create TensorMaps for outputs
    output_maps = []
    for tensor in output_tensors:
        name = "output"
        name_offset = builder.CreateString(name)
        TensorMap.TensorMapStart(builder)
        TensorMap.TensorMapAddName(builder, name_offset)
        TensorMap.TensorMapAddTensorIndex(builder, all_tensors.index(tensor))
        output_map = TensorMap.TensorMapEnd(builder)
        output_maps.append(output_map)

    # Create vectors of input and output TensorMaps

    SignatureDef.SignatureDefStartInputsVector(builder, len(input_maps))
    for input_map in reversed(input_maps):
        builder.PrependUOffsetTRelative(input_map)
    inputs_vector = builder.EndVector()

    SignatureDef.SignatureDefStartOutputsVector(builder, len(output_maps))
    for output_map in reversed(output_maps):
        builder.PrependUOffsetTRelative(output_map)
    outputs_vector = builder.EndVector()

    # Create the SignatureDef
    SignatureDef.SignatureDefStart(builder)
    SignatureDef.AddSignatureKey(builder, signature_key)
    SignatureDef.AddSubgraphIndex(builder, subgraph_index)
    SignatureDef.SignatureDefAddInputs(builder, inputs_vector)
    SignatureDef.SignatureDefAddOutputs(builder, outputs_vector)
    signature_def = SignatureDef.SignatureDefEnd(builder)

    return signature_def

def create_runtime_metadata(builder, buffers):
    #TODO - what metadata is there
    buffer = add_string_buffer(builder, buffers, "1.5.0")

    name = builder.CreateString("min_runtime_version")
    
    Metadata.Start(builder)
    Metadata.AddName(builder, name)
    Metadata.AddBuffer(builder, buffers.index(buffer))
    metadata = Metadata.End(builder)
    return metadata

def create_conversion_metadata(builder, buffers):
    #TODO - what metadata is there


    name = builder.CreateString("CONVERSION_METADATA")
    buffer = add_buffer(builder, buffers, jnp.array(len(buffers)-1))
    
    Metadata.Start(builder)
    Metadata.AddName(builder, name)
    Metadata.AddBuffer(builder, buffers.index(buffer))
    metadata = Metadata.End(builder)
    return metadata

def create_subgraph_tensors(builder, tensors):
    SubGraph.StartTensorsVector(builder, len(tensors))
    for tensor in reversed(tensors):
        builder.PrependUOffsetTRelative(tensor)
    subgraph_tensors = builder.EndVector()
    return subgraph_tensors

def create_operator_inputs(builder, input_tensors, all_tensors):
    Operator.StartInputsVector(builder, len(input_tensors))
    for itensor in reversed(input_tensors):
        if itensor not in all_tensors:
            builder.PrependInt32(-1) #if the target tensor doesn't exist, append -1 
        else:
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
    SubGraph.StartInputsVector(builder, len(input_tensors))
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

def create_subgraph(builder, subgraph_tensors, subgraph_inputs, subgraph_outputs, subgraph_ops, subgraph_name):

    #formalize builder constructs from python lists
    subgraph_name = builder.CreateString(subgraph_name)
    subgraph_inputs = create_subgraph_inputs(builder, subgraph_inputs, subgraph_tensors)
    subgraph_outputs = create_subgraph_outputs(builder, subgraph_outputs, subgraph_tensors)
    subgraph_tensors = create_subgraph_tensors(builder, subgraph_tensors)
    subgraph_ops = create_subgraph_ops(builder, subgraph_ops)

    SubGraph.Start(builder)
    SubGraph.AddName(builder, subgraph_name)
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

def create_metadatas(builder, metadatas):
    Model.StartMetadataVector(builder, len(metadatas) )
    for metadata in reversed(metadatas):
        builder.PrependUOffsetTRelative(metadata)
    metadatas = builder.EndVector()
    return metadatas

def create_signature_defs(builder, signature_defs):
    Model.StartSignatureDefsVector(builder, len(signature_defs) )
    for sig_def in reversed(signature_defs):
        builder.PrependUOffsetTRelative(sig_def)
    sig_defs = builder.EndVector()
    return sig_defs

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

def create_model(builder, subgraphs, op_codes, buffers, signature_defs, metadatas):
    metadatas = create_metadatas(builder, metadatas)
    signature_defs = create_signature_defs(builder, signature_defs)
    subgraphs = create_model_subgraphs(builder, subgraphs)
    op_codes = create_opcodes(builder, op_codes)
    buffers = create_buffers(builder, buffers)
    description = builder.CreateString("Quax Converted.")
    #description = builder.CreateString("MLIR Converted.")

    Model.Start(builder)
    Model.AddDescription(builder, description)
    Model.AddVersion(builder, 3)
    Model.AddSubgraphs(builder, subgraphs)
    Model.AddOperatorCodes(builder, op_codes)
    Model.AddBuffers(builder, buffers)
    Model.AddSignatureDefs(builder, signature_defs)
    Model.AddMetadata(builder, metadatas)
    model = Model.End(builder)
    return model

def export_tflite(builder, model):
    builder.Finish(model, file_identifier=_TFLITE_FILE_IDENTIFIER)
    tflite_model_data = builder.Output()
    return tflite_model_data



def map_jax_to_tflite_op(jax_primitive):
    jax_op_name = jax_primitive.name
    
    # Direct mappings
    direct_map = {
        'add': BuiltinOperator.ADD,
        'sub': BuiltinOperator.SUB,
        'mul': BuiltinOperator.MUL,
        'div': BuiltinOperator.DIV,
        'reduce_sum': BuiltinOperator.SUM,
        'reduce_max': BuiltinOperator.REDUCE_MAX,
        'reduce_min': BuiltinOperator.REDUCE_MIN,
        'reshape': BuiltinOperator.RESHAPE,
        'transpose': BuiltinOperator.TRANSPOSE,
        'slice': BuiltinOperator.SLICE,
        'concatenate': BuiltinOperator.CONCATENATION,
        'sin': BuiltinOperator.SIN,
        'cos': BuiltinOperator.COS,
        'tanh': BuiltinOperator.TANH,
        'exp': BuiltinOperator.EXP,
        'log': BuiltinOperator.LOG,
        'pow': BuiltinOperator.POW,
        'sqrt': BuiltinOperator.SQRT,
        'rsqrt': BuiltinOperator.RSQRT,
        'abs': BuiltinOperator.ABS,
        'neg': BuiltinOperator.NEG,
        'sign': BuiltinOperator.SIGN,
        'floor': BuiltinOperator.FLOOR,
        'ceil': BuiltinOperator.CEIL,
        'round': BuiltinOperator.ROUND,
        'squeeze': BuiltinOperator.SQUEEZE,
        'cast': BuiltinOperator.CAST,
    }
    
    if jax_op_name in direct_map:
        return direct_map[jax_op_name]
    
    # Special cases
    if jax_op_name == 'dot_general':
        # Note: This might need to be CONV_2D in some cases
        return BuiltinOperator.FULLY_CONNECTED
    elif jax_op_name == 'conv_general_dilated':
        return BuiltinOperator.CONV_2D
    elif jax_op_name == 'max_pool':
        return BuiltinOperator.MAX_POOL_2D
    elif jax_op_name == 'avg_pool':
        return BuiltinOperator.AVERAGE_POOL_2D
    elif jax_op_name == 'nn.relu':
        return BuiltinOperator.RELU
    elif jax_op_name == 'nn.sigmoid':
        return BuiltinOperator.LOGISTIC
    elif jax_op_name == 'nn.softmax':
        return BuiltinOperator.SOFTMAX
    elif jax_op_name == 'broadcast_in_dim':
        return BuiltinOperator.BROADCAST_TO
    elif jax_op_name == 'select':
        return BuiltinOperator.SELECT
    elif jax_op_name == 'gather':
        return BuiltinOperator.GATHER
    elif jax_op_name == 'dynamic_slice':
        return BuiltinOperator.SLICE  # Note: May need additional processing
    elif jax_op_name == 'dynamic_update_slice':
        return BuiltinOperator.DYNAMIC_UPDATE_SLICE
    elif jax_op_name == 'conv_transpose':
        return BuiltinOperator.TRANSPOSE_CONV
    elif jax_op_name == 'erf':
        # TFLite doesn't have a direct ERF op, might need to be approximated
        raise NotImplementedError("ERF operation not directly supported in TFLite")
    
    # If no mapping is found
    raise ValueError(f"Unsupported JAX primitive: {jax_op_name}")
