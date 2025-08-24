import flatbuffers
import numpy as np
# Using the new flatbuffer schema generated with object API
from quax.schema_py_generated import (
    Model, ModelT, SubGraph, SubGraphT, Tensor, TensorT, OperatorCode, OperatorCodeT,
    Buffer, BufferT, Operator, OperatorT, BuiltinOperator, 
    BuiltinOptions, FullyConnectedOptions, FullyConnectedOptionsT, ConcatenationOptions, ConcatenationOptionsT,
    ActivationFunctionType, AddOptions, AddOptionsT, MulOptions, MulOptionsT, TensorMap, TensorMapT,
    SignatureDef, SignatureDefT, Metadata, MetadataT, QuantizationParameters, QuantizationParametersT, 
    ReshapeOptions, ReshapeOptionsT, TensorType, Conv2DOptions, Conv2DOptionsT, Padding, 
    QuantizeOptions, QuantizeOptionsT, StridedSliceOptions, StridedSliceOptionsT, 
    SliceOptions, SliceOptionsT, DequantizeOptions, DequantizeOptionsT
)

from enum import Enum
import jax.numpy as jnp
import numpy as np
from quax.quax import AppendedActivation
_TFLITE_FILE_IDENTIFIER = b'TFL3'

def map_appended_activation(appended_activation):
    if appended_activation == None:
        return ActivationFunctionType.NONE
    act_map= {}
    act_map[AppendedActivation.RELU] = ActivationFunctionType.RELU
    act_map[AppendedActivation.RELU6] = ActivationFunctionType.TANH
    return act_map[appended_activation] 

def map_tensor_type(dtype):
    dtype_map = {}
    dtype_map[np.float32] = TensorType.FLOAT32
    dtype_map[np.int32] = TensorType.INT32
    dtype_map[np.int8] = TensorType.INT8
    dtype_map[np.int16] = TensorType.INT16
    dtype_map[np.int64] = TensorType.INT64
    return dtype_map[dtype]



def add_empty_tensor(tensor_name, tensor_dims, buffers, quantization_params=None, dtype=np.float32):
    # Create TensorT object using the object API
    tensor_obj = TensorT()
    tensor_obj.name = tensor_name
    tensor_obj.shape = list(tensor_dims)
    tensor_obj.type = map_tensor_type(dtype)
    tensor_obj.buffer = 0  # 0 buffer is standard notation for empty buffer (e.g. activations)
    tensor_obj.hasRank = True
    
    if quantization_params is None:
        quantization_params = add_empty_quant()
    tensor_obj.quantization = quantization_params
    
    return tensor_obj

def add_quantization_params(mins, maxs, scale, zero_point, quantized_dim):
    # Create QuantizationParametersT object using object API
    qparams_obj = QuantizationParametersT()
    
    # Convert vectors to numpy arrays 
    def to_array(x):
        if x is not None:
            return np.array(np.ravel(x))
        return None
    
    zero_point = np.array(zero_point, dtype=np.int64)
    qparams_obj.min = to_array(mins)
    qparams_obj.max = to_array(maxs)
    qparams_obj.scale = to_array(scale)
    qparams_obj.zeroPoint = to_array(zero_point)
    qparams_obj.quantizedDimension = quantized_dim
    qparams_obj.detailsType = 0
    
    return qparams_obj




def add_tensor_with_buffer(tensor_name, np_shape, buffer, buffers, dtype=np.float32):
    # Create TensorT object using the object API
    tensor_obj = TensorT()
    tensor_obj.name = tensor_name
    tensor_obj.shape = list(np_shape)
    tensor_obj.type = map_tensor_type(dtype)
    tensor_obj.hasRank = True
    
    # Find buffer index in list of BufferT objects
    buffer_idx = buffers.index(buffer)
    tensor_obj.buffer = buffer_idx
    
    # Add empty quantization
    tensor_obj.quantization = add_empty_quant()
    
    return tensor_obj

def add_empty_quant():
    # Create empty QuantizationParametersT object using object API
    quant_obj = QuantizationParametersT()
    # Return the object directly - it will be packed when the tensor is packed
    return quant_obj

#tensor add should have a numpy and string as input
def add_tensor(tensor_name, np_data, buffers, quantization_params=None, dtype=np.float32):
    #TODO - datatype resolution
    #TODO - type checking on input buffer and requested type
    np_shape = np_data.shape
    
    # Create buffer using object API
    buffer = add_buffer(buffers, np_data)
    buffers.append(buffer)
    buffer_idx = len(buffers) - 1

    # Create TensorT object using the object API
    tensor_obj = TensorT()
    tensor_obj.name = tensor_name
    tensor_obj.shape = list(np_shape)
    tensor_obj.type = map_tensor_type(dtype)
    tensor_obj.buffer = buffer_idx
    tensor_obj.hasRank = True
    
    if quantization_params is None:
        quantization_params = add_empty_quant()
    tensor_obj.quantization = quantization_params
    
    return tensor_obj

def get_empty_buffer():
    # Create empty BufferT object using object API
    buffer_obj = BufferT()
    return buffer_obj

def add_string_buffer(buffers, str_data):
    # Create BufferT object using object API
    buffer_obj = BufferT()
    # For string data, we need to convert to bytes
    buffer_obj.data = str_data.encode('utf-8')
    
    buffers.append(buffer_obj)
    return buffer_obj

def add_buffer(buffers, data=None):
    # Create BufferT object using object API
    buffer_obj = BufferT()
    if data is not None:
        buffer_obj.data = np.ravel(data).tobytes()
    
    return buffer_obj

def add_operator(inputs, outputs, options, options_type, opcode, all_opcodes):
    if opcode not in all_opcodes:
        all_opcodes.append(opcode)

    # Create OperatorT object using object API
    op_obj = OperatorT()
    op_obj.opcodeIndex = all_opcodes.index(opcode)
    op_obj.inputs = inputs
    op_obj.outputs = outputs
    if options:
        op_obj.builtinOptions = options
    if options_type:
        op_obj.builtinOptionsType = options_type
    
    return op_obj

def add_vec_layer(input_tensor1, input_tensor2, output_tensor, vec_op, all_tensors, all_opcodes):
    options = None 
    vec_options = None
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = vec_op
    opcode_obj.deprecatedBuiltinCode = vec_op
    
    # Create inputs and outputs as tensor indices
    inputs = [all_tensors.index(input_tensor1), all_tensors.index(input_tensor2)]
    outputs = [all_tensors.index(output_tensor)]

    add_op = add_operator(inputs, outputs, options, vec_options, opcode_obj, all_opcodes)
    return add_op

def add_concat_layer(input_tensors, output_tensor, axis, all_tensors, all_opcodes):
    # Create ConcatenationOptionsT object using object API
    concat_options_obj = ConcatenationOptionsT()
    concat_options_obj.axis = axis
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = BuiltinOperator.CONCATENATION
    
    concat_inputs = [all_tensors.index(t) for t in input_tensors]
    concat_outputs = [all_tensors.index(output_tensor)]
    concat_op = add_operator(concat_inputs, concat_outputs, concat_options_obj, BuiltinOptions.ConcatenationOptions, opcode_obj, all_opcodes)
    return concat_op

def add_mul_layer(input_tensor1, input_tensor2, output_tensor, all_tensors, all_opcodes):
    # Create MulOptionsT object using object API
    mul_options_obj = MulOptionsT()
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = BuiltinOperator.MUL
    
    # Create inputs and outputs
    mul_inputs = [all_tensors.index(input_tensor1), all_tensors.index(input_tensor2)]
    mul_outputs = [all_tensors.index(output_tensor)]

    mul_op = add_operator(mul_inputs, mul_outputs, mul_options_obj, BuiltinOptions.MulOptions, opcode_obj, all_opcodes)
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


def add_slice_layer(input_tensor, output_tensor,
                    slicing_key, all_tensors, all_opcodes, all_buffers):
    """
    Adds a Slice operator (begin / size, stride == 1) to the FlatBuffer model.

    Args:
        builder: flatbuffers.Builder instance.
        input_tensor: index of the tensor to slice.
        output_tensor: index of the tensor that will hold the slice.
        slicing_key: iterable of python slice objects (or ints for scalar-indexing).
                     Example: [slice(0, 1), slice(None, 193), slice(2, 10)]
        all_tensors / all_opcodes / all_buffers: running model lists.
    Returns:
        The created Slice operator.
    """
    begin = []
    size  = []

    for i, s in enumerate(slicing_key):
        if isinstance(s, int):
            begin.append(int(s))
            size.append(1)
        else:
            if isinstance(s, slice):
                # ----- begin -----
                start = s.start
                stop = s.stop
                step = s.step
            elif isinstance(s, tuple):
                start, stop, step = s
            b = 0 if start is None else start
            begin.append(b)

            # ----- size -----
            if stop is None:
                size.append(-1)        # -1 ⇒ “to the end” in TFLite
            else:
                size.append(stop - b)
            # ----- stride -----
            if step not in (None, 1):
                raise ValueError("Slice with step≠1 needs StridedSlice")


    # Constant tensors for begin / size
    for name, vec in (("slice_begin", begin), ("slice_size", size)):
        t_idx = add_tensor(
            name,
            np.asarray(vec, dtype=np.int32),
            all_buffers, dtype=np.int32)
        all_tensors.append(t_idx)

    op_inputs = [input_tensor, all_tensors[-2], all_tensors[-1]]

    slice_opcode = OperatorCodeT()
    slice_opcode.builtinCode = BuiltinOperator.SLICE


    slice_inputs  = [all_tensors.index(tensor) for tensor in op_inputs] 
    slice_outputs = [all_tensors.index(output_tensor)] 
    slice_op = add_operator(slice_inputs, slice_outputs,
                            None,  # options
                            None,
                            slice_opcode, all_opcodes)

    return slice_op
def add_strided_slice_layer(input_tensor, output_tensor, slicing_key, all_tensors, all_opcodes, all_buffers):
    """
    Adds a StridedSlice operator to the TFLite flatbuffer model.
    
    Args:
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
            start = s.start
            stop = s.stop
            step = s.step
        elif isinstance(s, tuple):
            #case - we converted (start,stop,step) to a tuple
            start, stop, step = s
        else:
            raise ValueError(f"Unsupported slicing type {type(s)} at dimension {i}")

        begin.append(0 if start is None else start)
        end.append(0 if stop is None else stop)
        strides.append(1 if step is None else step)
        
        # Adjust masks for None (default slicing)
        if start is None:
            begin_mask |= (1 << i)  # Ignore the begin value for this dimension
        if stop is None:
            end_mask |= (1 << i)    # Ignore the end value for this dimension

    # Create the `begin`, `end`, and `stride` tensors 
    op_inputs = [input_tensor]
    for idx,stride_elem in enumerate((begin, end, strides)):
        dtype = np.int32
        np_arr = np.array(stride_elem, dtype=dtype)
        in_tensor = add_tensor(f"stride_{idx}", np_arr, all_buffers, dtype=dtype)
        op_inputs.append(in_tensor)
        #TODO - a little messy to do things this way decentralized tensor accumulation
        all_tensors.append(in_tensor)

    strided_slice_options = StridedSliceOptionsT()

    strided_slice_opcode = OperatorCodeT()
    strided_slice_opcode.builtinCode = BuiltinOperator.STRIDED_SLICE

    # Create inputs and outputs
    strided_slice_inputs = [all_tensors.index(tensor) for tensor in op_inputs]
    strided_slice_outputs = [all_tensors.index(output_tensor)]


    # Add the StridedSlice operator
    strided_slice_op = add_operator(strided_slice_inputs, strided_slice_outputs, 
                                    strided_slice_options, BuiltinOptions.StridedSliceOptions, 
                                    strided_slice_opcode, all_opcodes)

    return strided_slice_op

def add_transpose_layer(input_tensor, output_tensor, axes_perm, all_buffers, all_tensors, all_opcodes):
    transpose_opcode = OperatorCodeT()
    transpose_opcode.builtinCode = BuiltinOperator.TRANSPOSE
    
    perm_tensor = add_tensor("perm", np.array(axes_perm,dtype=np.int32), all_buffers, dtype=np.int32)
    all_tensors.append(perm_tensor)
    # Create inputs and outputs
    transpose_inputs = [all_tensors.index(x) for x in [input_tensor, perm_tensor]]
    transpose_outputs = [all_tensors.index(x) for x in [output_tensor]]

    transpose_op = add_operator(transpose_inputs, transpose_outputs, None, None, transpose_opcode, all_opcodes)
    return transpose_op

def add_reshape_layer(input_tensor, output_tensor, new_shape, all_tensors, all_opcodes, all_buffers):
    # Create the ReshapeOptions
    reshape_options = ReshapeOptionsT()
    reshape_options.newShape = list(new_shape)
    op_inputs = [input_tensor]

    reshape_tensor = add_tensor(f"shape", np.array(new_shape, dtype=np.int32), all_buffers, dtype=np.int32)
    op_inputs.append(reshape_tensor)
    all_tensors.append(reshape_tensor)
    reshape_inputs = [all_tensors.index(tensor) for tensor in op_inputs]
    reshape_outputs = [all_tensors.index(output_tensor)] 
    reshape_opcode = OperatorCodeT()
    reshape_opcode.builtinCode = BuiltinOperator.RESHAPE

    reshape_op = add_operator(reshape_inputs, reshape_outputs, None,None, reshape_opcode, all_opcodes)
    return reshape_op

def add_conv_layer(input_tensor, weight_tensor, bias_tensor, output_tensor,bias_dtype,activation_op, all_tensors, all_opcodes,quax_params):

    conv_options = Conv2DOptionsT()

    conv_options.strideH = quax_params['window_strides'][0]
    conv_options.strideW = quax_params['window_strides'][1]

    conv_options.dilationHFactor = quax_params['rhs_dilation'][0]
    conv_options.dilationWFactor = quax_params['rhs_dilation'][1]

    conv_options.fusedActivationFunction = activation_op 

    if quax_params['padding'] == 'SAME':
        padding = Padding.SAME 
    else:
        padding = Padding.VALID 
    conv_options.padding =  padding

    conv_opcode = OperatorCodeT()
    conv_opcode.builtinCode = BuiltinOperator.CONV_2D
    
    #TODO - ordering here is fragile but important

    if bias_tensor:
        input_list = [input_tensor, weight_tensor, bias_tensor]
    else:
        input_list = [input_tensor, weight_tensor]
    conv_inputs = [all_tensors.index(tensor) for tensor in input_list]


    conv_outputs = [all_tensors.index(output_tensor)]

    conv_op = add_operator(conv_inputs, conv_outputs, conv_options, BuiltinOptions.Conv2DOptions, conv_opcode, all_opcodes)
    return conv_op

def add_activation_layer(input_tensor, output_tensor, operator_type, all_tensors, all_opcodes):
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = operator_type
    
    act_inputs = [all_tensors.index(input_tensor)]
    act_outputs = [all_tensors.index(output_tensor)]
    act_op = add_operator(act_inputs, act_outputs, None, None, opcode_obj, all_opcodes)
    return act_op

def add_dequant_layer(input_tensor, output_tensor, all_tensors, all_opcodes):
    # Create DequantizeOptionsT object using object API
    dequant_options_obj = DequantizeOptionsT()
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = BuiltinOperator.DEQUANTIZE
    
    #TODO - ordering here is fragile but important
    dequant_inputs = [all_tensors.index(input_tensor)]
    dequant_outputs = [all_tensors.index(output_tensor)]

    dequant_op = add_operator(dequant_inputs, dequant_outputs, dequant_options_obj, BuiltinOptions.DequantizeOptions, opcode_obj, all_opcodes)
    return dequant_op
def add_quant_layer(input_tensor, output_tensor, all_tensors, all_opcodes):
    # Create QuantizeOptionsT object using object API
    quant_options_obj = QuantizeOptionsT()
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = BuiltinOperator.QUANTIZE
    
    #TODO - ordering here is fragile but important
    quant_inputs = [all_tensors.index(input_tensor)]
    quant_outputs = [all_tensors.index(output_tensor)]

    quant_op = add_operator(quant_inputs, quant_outputs, quant_options_obj, BuiltinOptions.QuantizeOptions, opcode_obj, all_opcodes)
    return quant_op

def add_fc_layer(input_tensor, weight_tensor, bias_tensor, output_tensor, bias_dtype, activation_op, all_tensors, all_opcodes):
    # Create FullyConnectedOptionsT object using object API
    fc_options_obj = FullyConnectedOptionsT()
    fc_options_obj.fusedActivationFunction = activation_op
    
    # Create OperatorCodeT object using object API
    opcode_obj = OperatorCodeT()
    opcode_obj.builtinCode = BuiltinOperator.FULLY_CONNECTED
    
    #TODO - ordering here is fragile but important
    if bias_tensor:
        fc_inputs = [all_tensors.index(input_tensor), all_tensors.index(weight_tensor), all_tensors.index(bias_tensor)]
    else:
        fc_inputs = [all_tensors.index(input_tensor), all_tensors.index(weight_tensor)]
    fc_outputs = [all_tensors.index(output_tensor)]

    fc_op = add_operator(fc_inputs, fc_outputs, fc_options_obj, BuiltinOptions.FullyConnectedOptions, opcode_obj, all_opcodes)
    return fc_op


def create_signature_def(builder, input_tensors, output_tensors, all_tensors, subgraph_index):
    # Create TensorMaps for inputs
    signature_key = builder.CreateString("serving_default")
    input_maps = []
    for idx, tensor in enumerate(input_tensors):
        name = f"input_{idx}"
        name_offset = builder.CreateString(name)
        from quax.schema_py_generated import TensorMapStart, TensorMapAddName, TensorMapAddTensorIndex, TensorMapEnd
        TensorMapStart(builder)
        TensorMapAddName(builder, name_offset)
        TensorMapAddTensorIndex(builder, all_tensors.index(tensor))
        input_map = TensorMapEnd(builder)
        input_maps.append(input_map)

    # Create TensorMaps for outputs
    output_maps = []
    for idx, tensor in enumerate(output_tensors):
        name = f"output_{idx}"
        name_offset = builder.CreateString(name)
        TensorMapStart(builder)
        TensorMapAddName(builder, name_offset)
        TensorMapAddTensorIndex(builder, all_tensors.index(tensor))
        output_map = TensorMapEnd(builder)
        output_maps.append(output_map)

    # Create vectors of input and output TensorMaps
    from quax.schema_py_generated import SignatureDefStartInputsVector, SignatureDefStartOutputsVector, SignatureDefStart, SignatureDefAddSignatureKey, SignatureDefAddSubgraphIndex, SignatureDefAddInputs, SignatureDefAddOutputs, SignatureDefEnd

    SignatureDefStartInputsVector(builder, len(input_maps))
    for input_map in reversed(input_maps):
        builder.PrependUOffsetTRelative(input_map)
    inputs_vector = builder.EndVector()

    SignatureDefStartOutputsVector(builder, len(output_maps))
    for output_map in reversed(output_maps):
        builder.PrependUOffsetTRelative(output_map)
    outputs_vector = builder.EndVector()

    # Create the SignatureDef
    SignatureDefStart(builder)
    SignatureDefAddSignatureKey(builder, signature_key)
    SignatureDefAddSubgraphIndex(builder, subgraph_index)
    SignatureDefAddInputs(builder, inputs_vector)
    SignatureDefAddOutputs(builder, outputs_vector)
    signature_def = SignatureDefEnd(builder)

    return signature_def

def create_runtime_metadata(builder, buffers):
    #TODO - what metadata is there
    buffer = add_string_buffer(builder, buffers, "1.5.0")

    name = builder.CreateString("min_runtime_version")
    
    from quax.schema_py_generated import MetadataStart, MetadataAddName, MetadataAddBuffer, MetadataEnd
    MetadataStart(builder)
    MetadataAddName(builder, name)
    MetadataAddBuffer(builder, buffers.index(buffer))
    metadata = MetadataEnd(builder)
    return metadata

def create_conversion_metadata(builder, buffers):
    #TODO - what metadata is there

    name = builder.CreateString("CONVERSION_METADATA")
    buffer = add_buffer(builder, buffers, jnp.array(len(buffers)-1))
    
    from quax.schema_py_generated import MetadataStart, MetadataAddName, MetadataAddBuffer, MetadataEnd
    MetadataStart(builder)
    MetadataAddName(builder, name)
    MetadataAddBuffer(builder, buffers.index(buffer))
    metadata = MetadataEnd(builder)
    return metadata

def create_subgraph_tensors(builder, tensors):
    from quax.schema_py_generated import SubGraphStartTensorsVector
    SubGraphStartTensorsVector(builder, len(tensors))
    for tensor in reversed(tensors):
        builder.PrependUOffsetTRelative(tensor)
    subgraph_tensors = builder.EndVector()
    return subgraph_tensors

def create_operator_inputs(builder, input_tensors, all_tensors):
    raise Exception("deprecated")
    builder.StartVector(4, len(input_tensors), 4)
    for itensor in reversed(input_tensors):
        if itensor not in all_tensors:
            builder.PrependInt32(-1) #if the target tensor doesn't exist, append -1 
        else:
            builder.PrependInt32(all_tensors.index(itensor))  # input tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_operator_outputs(builder, output_tensors, all_tensors):
    raise Exception("deprecated")
    builder.StartVector(4, len(output_tensors), 4)
    for otensor in reversed(output_tensors):
        builder.PrependInt32(all_tensors.index(otensor))  # input tensor index
    subgraph_outputs = builder.EndVector()
    return subgraph_outputs


def create_subgraph_inputs(builder, input_tensors, all_tensors):
    from quax.schema_py_generated import SubGraphStartInputsVector
    SubGraphStartInputsVector(builder, len(input_tensors))
    for itensor in reversed(input_tensors):
        builder.PrependInt32(all_tensors.index(itensor))  # input tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_subgraph_outputs(builder, output_tensors, all_tensors):
    from quax.schema_py_generated import SubGraphStartOutputsVector
    SubGraphStartOutputsVector(builder, len(output_tensors))
    for otensor in reversed(output_tensors):
        builder.PrependInt32(all_tensors.index(otensor))  # output tensor index
    subgraph_inputs = builder.EndVector()
    return subgraph_inputs

def create_subgraph_ops(builder, ops):
    from quax.schema_py_generated import SubGraphStartOperatorsVector
    SubGraphStartOperatorsVector(builder, len(ops))
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

    from quax.schema_py_generated import SubGraphStart, SubGraphAddName, SubGraphAddTensors, SubGraphAddInputs, SubGraphAddOutputs, SubGraphAddOperators, SubGraphEnd
    
    SubGraphStart(builder)
    SubGraphAddName(builder, subgraph_name)
    SubGraphAddTensors(builder, subgraph_tensors)
    SubGraphAddInputs(builder, subgraph_inputs)
    SubGraphAddOutputs(builder, subgraph_outputs)
    SubGraphAddOperators(builder, subgraph_ops)
    subgraph = SubGraphEnd(builder)
    return subgraph

def create_model_subgraphs(builder, subgraphs):
    from quax.schema_py_generated import ModelStartSubgraphsVector
    ModelStartSubgraphsVector(builder, len(subgraphs))
    for subgraph in reversed(subgraphs):
        builder.PrependUOffsetTRelative(subgraph)
    subgraphs = builder.EndVector()
    return subgraphs

def create_metadatas(builder, metadatas):
    from quax.schema_py_generated import ModelStartMetadataVector
    ModelStartMetadataVector(builder, len(metadatas) )
    for metadata in reversed(metadatas):
        builder.PrependUOffsetTRelative(metadata)
    metadatas = builder.EndVector()
    return metadatas

def create_signature_defs(builder, signature_defs):
    from quax.schema_py_generated import ModelStartSignatureDefsVector
    ModelStartSignatureDefsVector(builder, len(signature_defs) )
    for sig_def in reversed(signature_defs):
        builder.PrependUOffsetTRelative(sig_def)
    sig_defs = builder.EndVector()
    return sig_defs

def create_opcodes(builder, op_codes):
    from quax.schema_py_generated import ModelStartOperatorCodesVector
    ModelStartOperatorCodesVector(builder, len(op_codes))
    for opcode in reversed(op_codes):
        builder.PrependUOffsetTRelative(opcode)
    op_codes = builder.EndVector()
    return op_codes

def create_buffers(builder, buffers):
    from quax.schema_py_generated import ModelStartBuffersVector
    ModelStartBuffersVector(builder, len(buffers))
    for buffer in reversed(buffers):
        builder.PrependUOffsetTRelative(buffer)
    buffers = builder.EndVector()
    return buffers

def create_model(builder, subgraphs, op_codes, buffers, signature_defs, metadatas):
    # Create ModelT object using object API
    model_obj = ModelT()
    model_obj.version = 3
    model_obj.description = "Quax Converted."
    
    # Convert the flatbuffer objects to object API equivalents
    # For now, we'll use the existing functions to create the vectors
    # A full refactor would convert these to use lists of T objects
    metadatas_vector = create_metadatas(builder, metadatas)
    signature_defs_vector = create_signature_defs(builder, signature_defs)
    subgraphs_vector = create_model_subgraphs(builder, subgraphs)
    op_codes_vector = create_opcodes(builder, op_codes)
    buffers_vector = create_buffers(builder, buffers)
    
    # For the object API, we would ideally set these as lists:
    # model_obj.metadata = metadata_objects  # List[MetadataT]
    # model_obj.signatureDefs = signature_def_objects  # List[SignatureDefT]
    # model_obj.subgraphs = subgraph_objects  # List[SubGraphT]
    # model_obj.operatorCodes = op_code_objects  # List[OperatorCodeT]
    # model_obj.buffers = buffer_objects  # List[BufferT]
    
    # For now, create model using traditional builder approach and pack
    from quax.schema_py_generated import ModelStart, ModelAddDescription, ModelAddVersion, ModelAddSubgraphs, ModelAddOperatorCodes, ModelAddBuffers, ModelAddSignatureDefs, ModelAddMetadata, ModelEnd
    
    # Create string before starting model building
    description_string = builder.CreateString(model_obj.description)
    
    ModelStart(builder)
    ModelAddDescription(builder, description_string)
    ModelAddVersion(builder, model_obj.version)
    ModelAddSubgraphs(builder, subgraphs_vector)
    ModelAddOperatorCodes(builder, op_codes_vector)
    ModelAddBuffers(builder, buffers_vector)
    ModelAddSignatureDefs(builder, signature_defs_vector)
    ModelAddMetadata(builder, metadatas_vector)
    model = ModelEnd(builder)
    return model

def create_runtime_metadata_obj(version_string: str, buffers: list) -> MetadataT:
    """Create runtime metadata object"""
    # Add version buffer
    buffer_obj = BufferT()
    buffer_obj.data = version_string.encode('utf-8')
    buffers.append(buffer_obj)
    
    metadata_obj = MetadataT()
    metadata_obj.name = "min_runtime_version"
    metadata_obj.buffer = len(buffers) - 1
    return metadata_obj

def create_conversion_metadata_obj(buffers: list) -> MetadataT:
    """Create conversion metadata object"""
    # Add conversion buffer
    buffer_obj = BufferT()
    buffer_obj.data = jnp.array(len(buffers)-1).tobytes()
    buffers.append(buffer_obj)
    
    metadata_obj = MetadataT()
    metadata_obj.name = "CONVERSION_METADATA"
    metadata_obj.buffer = len(buffers) - 1
    return metadata_obj

def export_tflite(model_obj: ModelT) -> bytes:
    """Serialize a ModelT object to TFLite flatbuffer format"""
    builder = flatbuffers.Builder(1024)
    
    # Pack the entire model object tree
    model = model_obj.Pack(builder)
    
    builder.Finish(model, file_identifier=_TFLITE_FILE_IDENTIFIER)
    return builder.Output()



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
